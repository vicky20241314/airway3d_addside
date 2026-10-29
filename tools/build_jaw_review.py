"""Chinese review page from measured, immutable paired-capture comparisons."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
root=ROOT/'diagnostics/jaw_candidate'
summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
sections=[]
for run,states in sorted(summary.items()):
    rows=[]
    for state,label in [('closed','闭口'),('open','张口')]:
        d=states[state];a=d['before'];b=d['after']
        for metric,x,y in [
            ('下颌轮廓法向偏差均值（px）',a['jaw_normal_px']['mean'],b['jaw_normal_px']['mean']),
            ('下颌带深度→表面 RMSE（mm）',a['surface']['jaw']['rmse_mm'],b['surface']['jaw']['rmse_mm']),
            ('其余面部深度→表面 RMSE（mm）',a['surface']['other_face']['rmse_mm'],b['surface']['other_face']['rmse_mm']),
            ('颈部深度→表面 RMSE（mm）',a['surface']['neck']['rmse_mm'],b['surface']['neck']['rmse_mm']),
            ('五官投影平均偏差（px）',a['internal_reprojection_px']['mean'],b['internal_reprojection_px']['mean'])]:
            rows.append(f'<tr><td>{label}</td><td>{metric}</td><td>{x:.3f}</td><td>{y:.3f}</td><td>{y-x:+.3f}</td></tr>')
    sections.append(f'''<section><h2>{run}</h2><table><tr><th>状态</th><th>指标</th><th>原版</th><th>下颌约束版</th><th>差值</th></tr>{''.join(rows)}</table>
<h3>闭口原图与 stage2 对比</h3><img src="{run}/closed/comparison.png">
<h3>张口原图与 stage2 对比</h3><img src="{run}/open/comparison.png">
<p><a href="{run}/closed/mesh_closed.obj">闭口最终 stage3 OBJ（米）</a> · <a href="{run}/open/mesh_open.obj">张口最终 stage3 OBJ（米）</a> · <a href="{run}/comparison.json">完整报告、输入哈希、选点及参数</a></p></section>''')
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Airway3D 下颌拟合改进</title><style>
body{font:16px/1.65 'Microsoft YaHei',sans-serif;color:#1c2834;background:#f2f5f8;max-width:1550px;margin:32px auto;padding:0 24px}section{background:white;border-radius:12px;padding:24px;margin:25px 0}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd5dd;padding:9px;text-align:left}img{width:100%}a{color:#1565c0}.note{padding:20px;background:#e4edf7}</style>
<h1>下颌轮廓偏圆：定位与改进</h1><p class="note">主要偏差已在 stage2 存在。新版本增加分割支持的下颌轮廓法向约束，并对下颌带／其余面部／颈部分区计权。保留共享 shape、原参数调度以及 stage3 的位移与几何保护。没有人为指定尖下巴，也没有使用扫描监督。</p>
<p>本页比较同一份 RGB-D 数据的重跑结果。每帧固定抽取 8000 个深度点，种子 42，区域由输入图像决定。深度贴合与二维轮廓是观测一致性指标，不是独立扫描误差或逐点解剖误差。三次闭口改善一致；部分张口下颌和颈部残差略增，详见完整表格。原文件未覆盖。</p>
'''+''.join(sections)+'''<p>默认流程版本：cross-v4-jaw-supported。明显侧脸或分割支持不足时回退。图片中的“候选”指本次独立对照输出，最终 stage3 可下载检查。</p></html>'''
(root/'index.html').write_text(html,encoding='utf-8')
print('Jaw comparison index.html written.')
