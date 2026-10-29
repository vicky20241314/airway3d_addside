"""Evaluate this cross-v1 closed run against VEGA on one fixed scan ROI.

The ROI and every mask are saved before any distances are computed. Both
versions are queried from the same scan surface samples, with no per-version
registration, error-based filtering, or reconstruction changes.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parents[1]
EVAL_PROJECT = Path(r"D:\项目\subdivide_scan")
PRIOR = EVAL_PROJECT / "jiaochastage3"
RUN = HERE / "validation/runs/20261024_194336_714226"
OUT = HERE / "validation/scan_evaluation_20261025"
SCAN = EVAL_PROJECT / "scan/scan-OBJ/scan.obj"
COUNT = 50000
SEED = 20261023

sys.path.insert(0, str(EVAL_PROJECT))
from evaluate_closed_scan import file_sha256, load_mesh  # noqa: E402
from evaluate_closed_pp import SurfaceDistance, summarize, write_json  # noqa: E402
from evaluate_five_versions import clip, footprint, polygons, visible_candidates  # noqa: E402
import scan_eval_figures as viz  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def statistics(distances):
    return {
        **summarize(distances),
        "mse_mm2": float(np.mean(distances**2)),
        "p99_mm": float(np.percentile(distances, 99)),
        "max_mm": float(np.max(distances)),
    }


def save_region_image(full_mesh, region_mesh, name, view):
    axes = viz.VIEWS[view]
    fig, ax = plt.subplots(figsize=(7, 7))
    viz.draw_surface(ax, full_mesh, axes, "#b8b8b8", 0.6)
    viz.draw_surface(ax, region_mesh, axes, "#08b965", 1)
    viz.setup_axes(ax, full_mesh, axes)
    ax.set_title(
        f"{name}｜共同面部区域｜{viz.VIEW_NAMES[view]}\n"
        f"绿色纳入，灰色排除；表面积 {region_mesh.area * 1e6:.1f} mm²"
    )
    fig.savefig(OUT / f"region_{name}_{view}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite an existing evaluation: {OUT}")

    audit = read(EVAL_PROJECT / "evalution_all_version_zhou/with_old_subdivision/input_audit.json")
    provenance = read(RUN / "provenance.json")
    assert provenance["capture_sha256"] == audit["capture_reference_sha256"]
    assert provenance["mesh_unit"] == "m"
    assert file_sha256(SCAN) == audit["inputs"]["scan"]["sha256"]

    matrix = np.loadtxt(PRIOR / "scan_to_reconstruction_m.txt")
    assert np.allclose(matrix[3], [0, 0, 0, 1])
    assert np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-8)
    assert np.allclose(matrix, np.asarray(audit["registration_matrix_m"]))

    paths = {
        "stage2": RUN / "closed/mesh_closed_stage2.obj",
        "stage3": RUN / "closed/mesh_closed.obj",
        "scan": SCAN,
    }
    meshes = {name: load_mesh(path, "mm" if name == "scan" else "m")
              for name, path in paths.items()}
    meshes["scan"].apply_transform(matrix)

    # Confirm that stage3 is the saved subdivision plus displacement of stage2.
    reference, faces = trimesh.remesh.subdivide(meshes["stage2"].vertices,
                                                meshes["stage2"].faces)
    with np.load(RUN / "closed/vertex_offsets.npz") as offsets:
        assert np.array_equal(faces, offsets["faces"])
        assert np.allclose(reference, offsets["reference_vertices"], atol=2e-8, rtol=0)
        assert np.allclose(offsets["reference_vertices"] + offsets["offsets"],
                           meshes["stage3"].vertices, atol=2e-8, rtol=0)

    prior_domain = shape(read(PRIOR / "common_region.geojson"))
    candidates = {name: visible_candidates(meshes[name], prior_domain)
                  for name in ("stage2", "stage3")}
    with np.load(PRIOR / "scan_mask.npz") as saved:
        candidates["scan"] = np.unique(saved["source_face_ids"])

    # Set the common XY face region before measuring any surface distances.
    domain = prior_domain
    for name in ("stage2", "stage3", "scan"):
        domain = domain.intersection(footprint(meshes[name], candidates[name]))
    domain = domain.buffer(0)
    domain = unary_union([piece for piece in polygons(domain) if piece.area >= 5])
    if domain.area < 10000:
        raise ValueError(f"Unexpectedly small shared frontal ROI: {domain.area} mm²")

    OUT.mkdir(parents=True)
    write_json(OUT / "common_region.geojson", mapping(domain))
    np.savetxt(OUT / "scan_to_reconstruction_m.txt", matrix)
    regions, region_info = {}, {}
    for name in ("stage2", "stage3", "scan"):
        region, parent, bary = clip(meshes[name], candidates[name], domain)
        projected = footprint(region, np.arange(len(region.faces))).area
        if abs(projected - domain.area) / domain.area > 1e-5:
            raise ValueError(f"Incomplete projected coverage: {name} {projected} vs {domain.area}")
        regions[name] = region
        np.savez_compressed(OUT / f"{name}_mask.npz", source_face_ids=parent,
                            barycentric=bary, clipped_vertices_m=region.vertices,
                            clipped_faces=region.faces)
        region.export(OUT / f"{name}_region.ply")
        region_info[name] = {
            "surface_area_mm2": float(region.area * 1e6),
            "projected_area_mm2": float(projected),
            "source_faces": int(len(np.unique(parent))),
            "clipped_triangles": int(len(region.faces)),
            "mask_sha256": file_sha256(OUT / f"{name}_mask.npz"),
        }
        for view in viz.VIEWS:
            save_region_image(meshes[name], region, name, view)
        print("REGION", name, region_info[name]["surface_area_mm2"], flush=True)

    region_rules = {
        "status": "prepared_before_distances",
        "method": "Previously approved seven-version frontal XY ROI intersected with visible coverage of both new reconstruction meshes and the aligned scan; no distance-based trimming",
        "source_geojson": str(PRIOR / "common_region.geojson"),
        "source_geojson_sha256": file_sha256(PRIOR / "common_region.geojson"),
        "projected_area_mm2": float(domain.area),
        "prior_projected_area_mm2": float(prior_domain.area),
        "meshes": region_info,
    }
    write_json(OUT / "regions.json", region_rules)
    print("PREPARED", domain.area, flush=True)

    scan_points, _ = trimesh.sample.sample_surface(regions["scan"], COUNT, seed=SEED)
    samples = {"scan_points_m": scan_points}
    results = {}
    for name in ("stage2", "stage3"):
        distances = SurfaceDistance(regions[name])(scan_points)
        samples[f"scan_to_{name}_mm"] = distances
        results[name] = statistics(distances)
        fig, ax = plt.subplots(figsize=(7, 7))
        plot = ax.scatter(scan_points[:, 0] * 1000, scan_points[:, 1] * 1000,
                          c=distances, s=0.4, cmap="turbo", vmin=0, vmax=10,
                          rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlabel("X（mm）")
        ax.set_ylabel("Y（mm）")
        ax.set_title(f"{name}｜扫描 → 重建\n同一批扫描采样点；色标 0–10 mm")
        fig.colorbar(plot, ax=ax, label="最近三角形表面距离（mm）", extend="max")
        fig.savefig(OUT / f"heatmap_scan_to_{name}.png", dpi=140,
                    bbox_inches="tight")
        plt.close(fig)
        print("METRIC", name, results[name], flush=True)
    np.savez_compressed(OUT / "samples.npz", **samples)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fields = ["mean_mm", "median_mm", "rmse_mm", "p95_mm"]
    x = np.arange(len(fields))
    for i, name in enumerate(("stage2", "stage3")):
        axes[0].bar(x + (i - 0.5) * 0.36, [results[name][field] for field in fields],
                    width=0.36, label=name)
    axes[0].set_xticks(x, ["均值", "中位数", "RMSE", "P95"])
    axes[0].set_ylabel("扫描 → 重建距离（mm）")
    axes[0].legend()
    bins = np.linspace(0, 15, 76)
    for name in ("stage2", "stage3"):
        axes[1].hist(samples[f"scan_to_{name}_mm"], bins=bins, density=True,
                     histtype="step", label=name)
    axes[1].set_xlabel("扫描 → 重建距离（mm）")
    axes[1].set_ylabel("概率密度（1/mm）")
    axes[1].legend()
    fig.suptitle(f"固定扫描区域；相同的 {COUNT:,} 个点；随机种子 {SEED}")
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=150)
    plt.close(fig)

    inputs = {name: {"path": str(path), "sha256": file_sha256(path),
                     "unit": "mm" if name == "scan" else "m"}
              for name, path in paths.items()}
    report = {
        "status": "complete",
        "inputs": inputs,
        "capture_sha256": provenance["capture_sha256"],
        "provenance_sha256": file_sha256(RUN / "provenance.json"),
        "offsets_sha256": file_sha256(RUN / "closed/vertex_offsets.npz"),
        "alignment_matrix_m": matrix.tolist(),
        "registration_source": str(PRIOR / "scan_to_reconstruction_m.txt"),
        "registration_source_sha256": file_sha256(PRIOR / "scan_to_reconstruction_m.txt"),
        "regions": region_rules,
        "regions_sha256": file_sha256(OUT / "regions.json"),
        "sampling": {"source": "aligned scan regional triangle surface",
                     "count": COUNT, "seed": SEED,
                     "method": "triangle-area uniform; identical scan points for both targets"},
        "metric": "Unfiltered scan-to-reconstruction nearest triangle surface distance in mm; not anatomical pointwise correspondence",
        "metrics": results,
        "stage3_minus_stage2": {field: results["stage3"][field] - results["stage2"][field]
                                  for field in results["stage2"]},
    }
    write_json(OUT / "report.json", report)
    html = '<meta charset="utf-8"><style>body{font:16px sans-serif;margin:24px;max-width:1100px}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:7px}</style>'
    html += '<h1>闭口网格与 VEGA 扫描：固定扫描面部区域</h1><p>相同扫描点 → stage2 / stage3 最近三角形表面距离。数值为毫米；不是逐点解剖学对应误差。</p>'
    html += '<p><a href="report.json">完整报告</a> · <a href="regions.json">区域规则与面积</a> · <a href="common_region.geojson">共同区域</a></p>'
    html += '<table><tr><th>版本</th><th>均值</th><th>中位数</th><th>RMSE</th><th>P95</th><th>MSE（mm²）</th></tr>'
    for name in ("stage2", "stage3"):
        value = results[name]
        html += f'<tr><td>{name}</td>' + ''.join(f'<td>{value[key]:.4f}</td>' for key in ["mean_mm", "median_mm", "rmse_mm", "p95_mm", "mse_mm2"]) + '</tr>'
    html += '</table><img src="comparison.png">'
    for name in ("stage2", "stage3", "scan"):
        html += f'<h2>{name}：绿色为纳入区域，灰色为排除区域</h2>'
        for view in viz.VIEWS:
            html += f'<img style="width:48%" src="region_{name}_{view}.png">'
        if name != "scan":
            html += f'<img src="heatmap_scan_to_{name}.png">'
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print("COMPLETE", OUT, flush=True)


if __name__ == "__main__":
    main()
