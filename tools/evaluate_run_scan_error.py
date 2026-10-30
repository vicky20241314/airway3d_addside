"""Bidirectional triangle-surface errors on a fixed, landmark-aligned scan ROI.

The old scan mask is transported using original triangle barycentrics. Its XY
polygon belongs to a different frame and must not be applied directly. Prepare
saves every mask and preview before evaluate computes any surface distance.
Reference utilities are loaded read-only from --reference-project.
"""

import argparse
import csv
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import shapely
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
import trimesh

HERE = Path(__file__).resolve().parents[1]
NAMES = ("stage2", "stage3", "scan")
LABELS = {"stage2": "stage2 拟合网格", "stage3": "stage3 最终重建", "scan": "扫描网格"}
FIELDS = ("mean_mm", "median_mm", "rmse_mm", "p95_mm", "p99_mm", "max_mm", "mse_mm2")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stamp():
    return datetime.now(timezone.utc).isoformat()


def source(path, unit=None):
    result = {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}
    if unit:
        result["unit"] = unit
    return result


def verify_sources(audit):
    for item in audit["sources"].values():
        if file_sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Input changed: {item['path']}")


def statistics(values, maximum):
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Invalid distances")
    return {**summarize(values), "sample_count": len(values),
            "mse_mm2": float(np.mean(values**2)), "p99_mm": float(np.percentile(values, 99)),
            "max_mm": float(values.max()),
            "within_threshold_percent": {f"{t:g}": float(100 * np.mean(values <= t))
                                         for t in sorted({0.5, 1, 2, 3, 5, 6, maximum})},
            "above_heatmap_max_percent": float(100 * np.mean(values > maximum))}


def transport_scan_mask(scan, prior, matrix):
    old_matrix = np.asarray(read(prior / "input_audit.json")["registration_matrix_m"])
    with np.load(prior / "scan_mask.npz", allow_pickle=False) as data:
        parents = data["source_face_ids"].astype(np.int64)
        bary = data["barycentric"].copy()
        saved = data["clipped_vertices_m"][data["clipped_faces"]]
    if bary.shape != (len(parents), 3, 3) or np.any(parents < 0) or np.any(parents >= len(scan.faces)):
        raise ValueError("Invalid prior scan mask indices")
    if not np.allclose(bary.sum(axis=2), 1, atol=1e-8, rtol=0) or bary.min() < -1e-6 or bary.max() > 1 + 1e-6:
        raise ValueError("Invalid prior scan barycentrics")
    raw = np.einsum("nij,njk->nik", bary, scan.triangles[parents])
    expected_old = raw @ old_matrix[:3, :3].T + old_matrix[:3, 3]
    error = float(np.max(np.abs(expected_old - saved)))
    if error > 1e-8:
        raise ValueError(f"Prior mask does not match original scan: {error} m")
    aligned = raw @ matrix[:3, :3].T + matrix[:3, 3]
    mesh = trimesh.Trimesh(aligned.reshape(-1, 3), np.arange(aligned.size // 3).reshape(-1, 3), process=False)
    return mesh, parents, bary, error


def region_png(full, region, name, view, output):
    axes = viz.VIEWS[view]
    fig, ax = plt.subplots(figsize=(7, 7))
    viz.draw_surface(ax, full, axes, "#b8b8b8", .5)
    viz.draw_surface(ax, region, axes, "#08b965", 1)
    viz.setup_axes(ax, full, axes)
    ax.set_title(f"{LABELS[name]} · {viz.VIEW_NAMES[view]}\n绿色为评估区域；面积 {region.area * 1e6:.1f} mm²")
    fig.savefig(output / f"region_{name}_{view}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def alignment_pngs(meshes, target, aligned, names, output, rmse):
    for view, axes in viz.VIEWS.items():
        fig, ax = plt.subplots(figsize=(8, 8))
        viz.draw_surface(ax, meshes["stage3"], axes, viz.BLUE, .3)
        viz.draw_surface(ax, meshes["scan"], axes, viz.ORANGE, .25)
        viz.setup_axes(ax, meshes["stage3"], axes)
        p, q = target[:, list(axes[:2])] * 1000, aligned[:, list(axes[:2])] * 1000
        ax.scatter(*p.T, c=viz.BLUE, s=28, label="重建标点", zorder=4)
        ax.scatter(*q.T, c=viz.ORANGE, marker="x", s=40, label="扫描标点", zorder=4)
        for name, a, b in zip(names, p, q):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="crimson", lw=1, zorder=3)
            ax.annotate("R" + name, a, xytext=(4, 5), textcoords="offset points", fontsize=9)
            ax.annotate("S" + name, b, xytext=(4, -10), textcoords="offset points", fontsize=9)
        ax.legend(loc="lower left")
        ax.set_title(f"标点刚性配准 · {viz.VIEW_NAMES[view]}\n标点 RMSE {rmse:.3f} mm；红线为残差")
        fig.savefig(output / f"alignment_stage3_{view}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


def prepare(args):
    output = args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    paths = {"stage2": args.run_dir / "closed/mesh_closed_stage2.obj",
             "stage3": args.run_dir / "closed/mesh_closed.obj", "scan": args.scan,
             "reconstruction_pp": args.reconstruction_pp, "scan_pp": args.scan_pp,
             "prior_scan_mask": args.prior_dir / "scan_mask.npz",
             "prior_input_audit": args.prior_dir / "input_audit.json",
             "prior_domain": args.prior_dir / "common_region.geojson",
             "offsets": args.run_dir / "closed/vertex_offsets.npz",
             "provenance": args.run_dir / "provenance.json", "evaluation_script": Path(__file__).resolve()}
    for module in ("evaluate_closed_scan", "evaluate_closed_pp", "evaluate_five_versions", "scan_eval_figures"):
        paths["reference_" + module] = args.reference_project / (module + ".py")
    sources = {n: source(p, "mm" if n == "scan" else "m" if n in NAMES else None) for n, p in paths.items()}
    if sources["scan"]["sha256"] != read(paths["prior_input_audit"])["inputs"]["scan"]["sha256"]:
        raise ValueError("Prior mask belongs to a different scan")
    meshes = {n: load_mesh(paths[n], "mm" if n == "scan" else "m") for n in NAMES}
    names, target, points, meta = paired_pp(args.reconstruction_pp, args.scan_pp,
        paths["stage3"], paths["scan"], "m", "mm")
    matrix, residuals, aligned = rigid_scan_to_stage2(points, target)
    residuals_mm = residuals * 1000
    rmse = float(np.sqrt(np.mean(residuals_mm**2)))
    alerts = [f"标点 {n} 残差 {v:.3f} mm，超过 5 mm；保留该点。" for n, v in zip(names, residuals_mm) if v > 5]
    for n, c in meta["conditioning"].items():
        if c["third_to_first_ratio"] < .01:
            alerts.append(f"{n} 标点接近同一平面（第三/第一奇异值比 {c['third_to_first_ratio']:.3%}），深度方向标点覆盖有限。")
    alerts.append("表面误差包含当前标点配准的影响，不能完全归因于重建；不是逐点解剖学对应误差。")
    vertices, faces = trimesh.remesh.subdivide(meshes["stage2"].vertices, meshes["stage2"].faces)
    with np.load(paths["offsets"], allow_pickle=False) as data:
        if not np.array_equal(faces, data["faces"]) or not np.array_equal(faces, meshes["stage3"].faces):
            raise ValueError("Stage3 faces do not match subdivision/offsets")
        ref_error = float(np.max(np.abs(vertices - data["reference_vertices"])))
        final_error = float(np.max(np.abs(data["reference_vertices"] + data["offsets"] - meshes["stage3"].vertices)))
        moves = np.linalg.norm(data["offsets"], axis=1) * 1000
        if ref_error > 2e-8 or final_error > 2e-8:
            raise ValueError("Stage3 does not match subdivided stage2 plus offsets")
        geometry = {"subdivision_max_abs_error_m": ref_error, "offsets_max_abs_error_m": final_error,
                    "offset_mean_mm": float(moves.mean()), "offset_max_mm": float(moves.max())}
    prior_scan, scan_parents, scan_bary, recovery_error = transport_scan_mask(meshes["scan"], args.prior_dir, matrix)
    meshes["scan"].apply_transform(matrix)
    initial = footprint(prior_scan, np.arange(len(prior_scan.faces)))
    candidates = {n: visible_candidates(meshes[n], initial) for n in ("stage2", "stage3")}
    domain, coverage = initial, {"transported_prior_scan": float(initial.area)}
    for n in ("stage2", "stage3"):
        domain = domain.intersection(footprint(meshes[n], candidates[n]))
        coverage[n] = float(domain.area)
    domain = domain.buffer(-args.boundary_erosion_mm).buffer(0)
    domain = unary_union([p for p in polygons(domain) if p.area >= 5])
    if domain.is_empty or domain.area < 5000:
        raise ValueError(f"Unexpectedly small ROI: {domain.area} mm²")
    output.mkdir(parents=True, exist_ok=False)
    np.savetxt(output / "scan_to_reconstruction_m.txt", matrix, fmt="%.17g")
    meshes["scan"].export(output / "aligned_scan.ply")
    write_json(output / "transported_prior_scan_region.geojson", mapping(initial))
    write_json(output / "common_region.geojson", mapping(domain))
    regions, info = {}, {}
    for n in NAMES:
        if n == "scan":
            region, parent, bary = clip(prior_scan, np.arange(len(prior_scan.faces)), domain)
            bary = np.einsum("nij,njk->nik", bary, scan_bary[parent])
            parent = scan_parents[parent]
        else:
            region, parent, bary = clip(meshes[n], candidates[n], domain)
        projected = footprint(region, np.arange(len(region.faces))).area
        if abs(projected - domain.area) / domain.area > 1e-5:
            raise ValueError(f"Incomplete projected coverage: {n}")
        error = float(np.max(np.abs(np.einsum("nij,njk->nik", bary, meshes[n].triangles[parent]) - region.triangles)))
        if error > 1e-9:
            raise ValueError(f"Clipping moved original surface: {n}")
        np.savez_compressed(output / f"{n}_mask.npz", source_face_ids=parent, barycentric=bary,
                            clipped_vertices_m=region.vertices, clipped_faces=region.faces)
        region.export(output / f"{n}_region.ply")
        regions[n] = region
        info[n] = {"surface_area_mm2": float(region.area * 1e6), "projected_area_mm2": float(projected),
                   "fraction_of_full_mesh_area": float(region.area / meshes[n].area),
                   "source_faces": int(len(np.unique(parent))), "clipped_triangles": len(region.faces),
                   "max_barycentric_reconstruction_error_m": error,
                   "mask_sha256": file_sha256(output / f"{n}_mask.npz")}
        for view in viz.VIEWS:
            region_png(meshes[n], region, n, view, output)
        print("REGION", n, json.dumps(info[n]), flush=True)
    alignment_pngs(meshes, target, aligned, names, output, rmse)
    with (output / "landmark_residuals.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["name", "residual_mm"])
        writer.writerows(zip(names, residuals_mm))
    rules = {"status": "prepared_before_distances", "prepared_at_utc": stamp(),
             "method": "Exact prior scan mask transported by original scan triangle barycentrics into current landmark-aligned frame; intersected with stage2/stage3 visible XY coverage; no error-based selection",
             "coordinate_system": "Current reconstruction frame, XY in mm; all saved 3D meshes in m",
             "prior_geojson_used_directly_in_current_frame": False,
             "prior_domain_area_mm2": float(shape(read(paths["prior_domain"])).area),
             "prior_mask_recovery_max_abs_error_m": recovery_error,
             "coverage_before_erosion_mm2": coverage, "boundary_erosion_mm": args.boundary_erosion_mm,
             "minimum_component_projected_area_mm2": 5,
             "visibility": "Reconstruction normal Z > 0.05 and first orthographic ray hit at triangle centroid; scan inherits prior face selection",
             "projected_area_mm2": float(domain.area), "meshes": info,
             "area_comparability": "Identical XY support; surface area may differ with slope; not anatomical correspondence",
             "no_surface_distances_computed_at_preparation": True}
    audit = {"sources": sources, "geometry_audit": geometry,
             "mesh_geometry": {n: {"vertices": len(m.vertices), "faces": len(m.faces), "bounds_m": m.bounds.tolist(),
                                    "surface_area_mm2": float(m.area * 1e6), "watertight": bool(m.is_watertight)} for n, m in meshes.items()},
             "registration": {"method": "All active PP pairs; proper Kabsch, scale=1; no ICP", "source": "scan",
                              "target": "stage3 PP landmarks", "matrix_m": matrix.tolist(),
                              "determinant": float(np.linalg.det(matrix[:3, :3])), "same_transform_for_both_versions": True},
             "landmarks": {**meta, "names": names, "target_points_m": target.tolist(), "raw_scan_points_m": points.tolist(),
                           "aligned_scan_points_m": aligned.tolist(), "residuals_mm": residuals_mm.tolist(),
                           "rmse_mm": rmse, "removed_points": [], "warnings": alerts},
             "configuration": config(args),
             "software": {"python": sys.version.split()[0], "numpy": np.__version__, "trimesh": trimesh.__version__,
                          "shapely": shapely.__version__, "matplotlib": matplotlib.__version__}}
    verify_sources(audit)
    write_json(output / "regions.json", rules)
    write_json(output / "input_audit.json", audit)
    cards = ''.join(f'<h2>{LABELS[n]}</h2><img src="region_{n}_front.png"><img src="region_{n}_side.png">' for n in NAMES)
    (output / "review.html").write_text('<meta charset="utf-8"><style>body{font:16px system-ui;margin:24px}img{width:48%}</style>'
        '<h1>距离计算前的区域检查</h1><p>绿色纳入；灰色排除。区域由扫描面片和可见覆盖确定，不根据距离删点。</p>'
        + cards + '<h2>标点配准</h2><img src="alignment_stage3_front.png"><img src="alignment_stage3_side.png">', encoding="utf-8")
    print("PREPARED", output, "projected_area_mm2", domain.area, "landmark_rmse_mm", rmse, flush=True)


def heatmaps(regions, errors, samples, args):
    norm = Normalize(0, args.heatmap_max_mm)
    for n in ("stage2", "stage3"):
        region = regions[n]
        face_values = errors[n][region.faces].mean(axis=1)
        colored = region.copy()
        colored.visual.vertex_colors = (plt.get_cmap("turbo")(norm(errors[n])) * 255).astype(np.uint8)
        colored.export(args.output / f"{n}_heatmap.ply")
        for view, axes in viz.VIEWS.items():
            fig, ax = plt.subplots(figsize=(8, 8))
            viz.draw_surface(ax, region, axes, None, face_values=face_values, norm=norm)
            viz.setup_axes(ax, regions["stage2"], axes)
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="turbo"), ax=ax,
                         label="重建 → 扫描的最近三角面距离（mm）", extend="max")
            ax.set_title(f"{LABELS[n]} · {viz.VIEW_NAMES[view]}\n色标 0–{args.heatmap_max_mm:g} mm；超限饱和显示，统计保留")
            fig.savefig(args.output / f"heatmap_{n}_{view}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        points = samples["scan_points_m"]
        fig, ax = plt.subplots(figsize=(8, 8))
        plot = ax.scatter(points[:, 0] * 1000, points[:, 1] * 1000, c=samples[f"scan_to_{n}_mm"],
                          s=.6, cmap="turbo", norm=norm, rasterized=True)
        viz.setup_axes(ax, regions["stage2"], viz.VIEWS["front"])
        fig.colorbar(plot, ax=ax, label="扫描 → 重建的最近三角面距离（mm）", extend="max")
        ax.set_title(f"扫描 → {n}（辅助）\n相同 {args.samples:,} 个扫描点；色标 0–{args.heatmap_max_mm:g} mm")
        fig.savefig(args.output / f"heatmap_scan_to_{n}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def comparison(metrics, samples, args):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(4)
    for row, direction in enumerate(("reconstruction_to_scan", "scan_to_reconstruction")):
        for i, n in enumerate(("stage2", "stage3")):
            bars = axes[row, 0].bar(x + (i - .5) * .36, [metrics[direction][n][k] for k in FIELDS[:4]],
                width=.36, label=n, color=(viz.BLUE, viz.ORANGE)[i])
            axes[row, 0].bar_label(bars, fmt="%.3f", fontsize=9)
            distances = np.sort(samples[f"{n}_to_scan_mm" if row == 0 else f"scan_to_{n}_mm"])
            axes[row, 1].plot(distances, np.arange(1, len(distances) + 1) / len(distances) * 100,
                              label=n, color=(viz.BLUE, viz.ORANGE)[i])
        axes[row, 0].set_xticks(x, ["均值", "中位数", "RMSE", "P95"])
        axes[row, 0].set_ylabel("最近三角面距离（mm）")
        axes[row, 0].set_ylim(0, max(metrics[direction][n]["p95_mm"] for n in ("stage2", "stage3")) * 1.2)
        axes[row, 0].legend()
        axes[row, 0].set_title(("主指标：重建 → 扫描", "辅助：相同扫描点 → 重建")[row])
        axes[row, 1].axvline(args.heatmap_max_mm, c="gray", ls="--", lw=1, label=f"{args.heatmap_max_mm:g} mm")
        axes[row, 1].set_xlabel("最近三角面距离（mm）；显示全部采样范围")
        axes[row, 1].set_ylabel("累计采样占比（%）")
        axes[row, 1].set_title("完整距离累积分布；不截断超限值")
        axes[row, 1].grid(alpha=.2)
        axes[row, 1].legend(loc="lower right")
    fig.suptitle(f"固定面部区域 · 每表面 {args.samples:,} 点 · 种子 {args.seed}\n两版共用一次标点刚性配准；全部距离参与统计")
    fig.tight_layout()
    fig.savefig(args.output / "comparison.png", dpi=150)
    plt.close(fig)


def reports(report, args):
    metrics, rows = report["metrics"], []
    header = ["方向", "版本", "均值 mm", "中位数 mm", "RMSE mm", "P95 mm", "P99 mm", "采样最大值 mm", "MSE mm²", "≤2 mm %", "≤6 mm %"]
    for direction, label in (("reconstruction_to_scan", "重建→扫描（主）"), ("scan_to_reconstruction", "扫描→重建（辅助）")):
        for n in ("stage2", "stage3"):
            v = metrics[direction][n]
            rows.append([label, n, *[f"{v[k]:.4f}" for k in FIELDS],
                         *[f"{v['within_threshold_percent'][str(t)]:.2f}" for t in (2, 6)]])
    delta = report["stage3_minus_stage2"]["reconstruction_to_scan"]
    notes = [
        "评估固定共同面部区域内的无符号最近三角面距离；不是全头误差或逐点解剖学对应误差。距离越小表示在当前区域和配准下越接近扫描。",
        f"stage3 − stage2：主方向均值差 {delta['mean_mm']:+.4f} mm，RMSE 差 {delta['rmse_mm']:+.4f} mm。未给出临床合格阈值。",
        f"共同 XY 投影面积 {report['regions']['projected_area_mm2']:.2f} mm²，边界内缩 {args.boundary_erosion_mm:g} mm。",
        f"每表面按三角形面积均匀采样 {args.samples:,} 点，种子 {args.seed}。两版主方向各自采样；辅助方向复用完全相同的扫描点，可作配对比较。",
        f"热力图统一色标 0–{args.heatmap_max_mm:g} mm。超限仅饱和显示，完整保留在 samples.npz 并计入所有统计；不按误差删点。",
        "三角面颜色使用三个顶点距离的平均值，仅供显示；统计使用面积均匀采样距离。采样最大值不是精确 Hausdorff 距离。",
        "旧扫描区域经原始三角面及重心坐标转入本次坐标系，与两版可见覆盖求交，在任何表面距离计算前保存。两版共用当前标点求出的单次刚性配准，无缩放、无 ICP、无逐版重新配准。",
        f"标点 RMSE：{report['landmarks']['rmse_mm']:.4f} mm。", *report["landmarks"]["warnings"]]
    lines = ["# 重建网格与扫描网格误差评估", "", f"运行目录：`{args.run_dir}`", "",
             "|" + "|".join(header) + "|", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["|" + "|".join(row) + "|" for row in rows]
    lines += [""] + ["- " + note for note in notes] + ["", "文件：index.html 中文图册；report.json 完整统计；metrics.csv 统计表；samples.npz 原始点和距离（坐标 m，距离 mm）；*_heatmap.ply 彩色网格；regions.json / common_region.geojson / *_mask.npz 固定区域；input_audit.json 输入与算法 SHA-256；aligned_scan.ply / scan_to_reconstruction_m.txt / landmark_residuals.csv 配准证据。", ""]
    (args.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["direction", "version", *FIELDS, "within_2mm_percent", "within_6mm_percent"])
        for d, versions in metrics.items():
            for n, v in versions.items():
                writer.writerow([d, n, *[v[k] for k in FIELDS], v["within_threshold_percent"]["2"], v["within_threshold_percent"]["6"]])
    page = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>网格误差评估</title>',
            '<style>body{font:16px system-ui;margin:24px;max-width:1250px;color:#20334a}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #ccd3dc;padding:8px;text-align:right}.pair img{width:49%}.alert{background:#fff4df;padding:12px;line-height:1.7}</style>',
            '<h1>重建网格与扫描网格误差评估</h1><p><a href="summary.md">评估说明</a> · <a href="report.json">完整 JSON</a> · <a href="metrics.csv">统计 CSV</a> · <a href="review.html">计算前区域检查</a></p>',
            '<table><tr>' + ''.join('<th>' + html.escape(c) + '</th>' for c in header) + '</tr>']
    page += ['<tr>' + ''.join('<td>' + html.escape(c) + '</td>' for c in row) + '</tr>' for row in rows]
    page += ['</table><div class="alert">' + '<br>'.join(html.escape(n) for n in notes) + '</div><img src="comparison.png">']
    for n in ("stage3", "stage2"):
        page += [f'<h2>{LABELS[n]}：重建→扫描热图</h2><div class="pair"><img src="heatmap_{n}_front.png"><img src="heatmap_{n}_side.png"></div><p><a href="{n}_heatmap.ply">彩色 PLY 网格（m）</a></p>']
    page += ['<h2>辅助方向：相同扫描点→重建</h2><div class="pair"><img src="heatmap_scan_to_stage2.png"><img src="heatmap_scan_to_stage3.png"></div>']
    for n in NAMES:
        page += [f'<h2>{LABELS[n]}：绿色为纳入区域</h2><div class="pair"><img src="region_{n}_front.png"><img src="region_{n}_side.png"></div>']
    page += ['<h2>配准检查</h2><div class="pair"><img src="alignment_stage3_front.png"><img src="alignment_stage3_side.png"></div></html>']
    (args.output / "index.html").write_text("\n".join(page), encoding="utf-8")


def config(args):
    return {"samples_per_surface": args.samples, "seed": args.seed,
            "heatmap_max_mm": args.heatmap_max_mm, "boundary_erosion_mm": args.boundary_erosion_mm}


def evaluate(args):
    if (args.output / "report.json").exists() or (args.output / "samples.npz").exists():
        raise FileExistsError("Samples/results already exist; preserve them")
    audit, rules = read(args.output / "input_audit.json"), read(args.output / "regions.json")
    verify_sources(audit)
    if audit["configuration"] != config(args):
        raise ValueError("Arguments differ from preparation")
    regions = {}
    for n in NAMES:
        if file_sha256(args.output / f"{n}_mask.npz") != rules["meshes"][n]["mask_sha256"]:
            raise ValueError(f"Prepared mask changed: {n}")
        with np.load(args.output / f"{n}_mask.npz", allow_pickle=False) as data:
            regions[n] = trimesh.Trimesh(data["clipped_vertices_m"], data["clipped_faces"], process=False)
    samples, metrics, errors = {}, {"reconstruction_to_scan": {}, "scan_to_reconstruction": {}}, {}
    scan_points, scan_faces = trimesh.sample.sample_surface(regions["scan"], args.samples, seed=args.seed)
    samples.update(scan_points_m=scan_points, scan_region_face_ids=scan_faces)
    scan_query = SurfaceDistance(regions["scan"])
    for n in ("stage2", "stage3"):
        points, faces = trimesh.sample.sample_surface(regions[n], args.samples, seed=args.seed)
        distances = scan_query(points)
        samples.update({f"{n}_points_m": points, f"{n}_region_face_ids": faces, f"{n}_to_scan_mm": distances})
        metrics["reconstruction_to_scan"][n] = statistics(distances, args.heatmap_max_mm)
        values = SurfaceDistance(regions[n])(scan_points)
        samples[f"scan_to_{n}_mm"] = values
        metrics["scan_to_reconstruction"][n] = statistics(values, args.heatmap_max_mm)
        vertices, inverse = np.unique(regions[n].vertices, axis=0, return_inverse=True)
        errors[n] = scan_query(vertices)[inverse]
        samples[f"{n}_vertex_distance_mm"] = errors[n]
        print("METRIC", n, json.dumps({d: metrics[d][n] for d in metrics}), flush=True)
    verify_sources(audit)
    np.savez_compressed(args.output / "samples.npz", **samples)
    heatmaps(regions, errors, samples, args)
    comparison(metrics, samples, args)
    report = {"status": "complete", "completed_at_utc": stamp(), **audit,
              "regions": rules, "regions_sha256": file_sha256(args.output / "regions.json"),
              "sampling": {"method": "Triangle-area uniform, separately for each source surface", "samples_per_surface": args.samples,
                           "seed": args.seed, "same_scan_points_for_both_targets": True, "primary_samples_are_not_anatomical_pairs": True},
              "metric": "Unsigned nearest-triangle distance in mm on saved common frontal ROI; all distances included", "metrics": metrics,
              "stage3_minus_stage2": {d: {k: metrics[d]["stage3"][k] - metrics[d]["stage2"][k] for k in FIELDS} for d in metrics},
              "paired_scan_comparison": {"mean_distance_change_mm": float(np.mean(samples["scan_to_stage3_mm"] - samples["scan_to_stage2_mm"])),
                                         "fraction_points_improved_percent": float(100 * np.mean(samples["scan_to_stage3_mm"] < samples["scan_to_stage2_mm"]))},
              "heatmap": {"minimum_mm": 0, "maximum_mm": args.heatmap_max_mm, "above_max": "Display saturation only; included in statistics",
                          "face_color": "Mean vertex distances, display only", "same_limits_and_views": True},
              "samples_sha256": file_sha256(args.output / "samples.npz"), "input_files_unchanged": True}
    reports(report, args)
    verify_sources(audit)
    write_json(args.output / "report.json", report)
    print("COMPLETE", args.output, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("prepare", "evaluate", "all"), default="all")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--reference-project", type=Path, default=HERE.parent / "subdivide_scan")
    for n in ("prior-dir", "scan", "reconstruction-pp", "scan-pp", "output"):
        p.add_argument("--" + n, type=Path)
    p.add_argument("--samples", type=int, default=50000)
    p.add_argument("--seed", type=int, default=20261023)
    p.add_argument("--heatmap-max-mm", type=float, default=6)
    p.add_argument("--boundary-erosion-mm", type=float, default=.5)
    args = p.parse_args()
    if args.samples < 1 or args.seed < 0 or not np.isfinite(args.heatmap_max_mm) or args.heatmap_max_mm <= 0:
        raise ValueError("Invalid sample count, seed or heatmap maximum")
    if not np.isfinite(args.boundary_erosion_mm) or args.boundary_erosion_mm < 0:
        raise ValueError("Invalid boundary erosion")
    evaluation = args.run_dir / "evalution"
    args.prior_dir = args.prior_dir or args.reference_project / "evalution/evalution_all_version_zhou/with_old_subdivision"
    args.scan = args.scan or evaluation / "scan/scan-OBJ/scan.obj"
    args.reconstruction_pp = args.reconstruction_pp or evaluation / "mesh_closed_picked_points.pp"
    args.scan_pp = args.scan_pp or evaluation / "scan_picked_points.pp"
    args.output = args.output or evaluation / "surface_error_fixed_roi_6mm"
    sys.path.insert(0, str(args.reference_project))
    global file_sha256, load_mesh, rigid_scan_to_stage2, paired_pp, SurfaceDistance, summarize, write_json
    global clip, footprint, polygons, visible_candidates, viz
    from evaluate_closed_scan import file_sha256, load_mesh, rigid_scan_to_stage2
    from evaluate_closed_pp import paired_pp, SurfaceDistance, summarize, write_json
    from evaluate_five_versions import clip, footprint, polygons, visible_candidates
    import scan_eval_figures as viz
    if args.phase in ("prepare", "all"):
        prepare(args)
    if args.phase in ("evaluate", "all"):
        evaluate(args)


if __name__ == "__main__":
    main()
