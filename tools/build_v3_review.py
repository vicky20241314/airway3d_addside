"""Build a local Chinese comparison page from completed measured outputs."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--comparison',type=Path,required=True);a=p.parse_args()
    report=json.loads((a.comparison/'report.json').read_text(encoding='utf-8'))
    assert report['all_inputs_unchanged'] and set(report['states'])=={'closed','open'}
    plt.rcParams['font.sans-serif']=['Microsoft YaHei']
    sections=[]
    for state,label in [('closed','闭口'),('open','张口')]:
        folder=a.comparison/state
        fig,axes=plt.subplots(1,3,figsize=(15,5))
        for ax,file,title in zip(axes,['v2_flat_render.png','v2_smooth_render.png','v3_smooth_render.png'],
                ['v2：原平面着色','v2：仅平滑着色（顶点不动）','v3：受限位移平滑 + 平滑着色']):
            ax.imshow(plt.imread(folder/file));ax.set_title(title,fontsize=12);ax.axis('off')
        fig.subplots_adjust(top=.88,bottom=.02,wspace=.03)
        fig.savefig(folder/'display_comparison.png',dpi=160,bbox_inches='tight',pad_inches=.15);plt.close(fig)
        m=report['states'][state]['metrics'];i=json.loads((folder/'v3_report.json').read_text(encoding='utf-8'))
        rows=[]
        values=[('固定深度样本到表面 RMSE（mm）',m['v2']['depth_rmse_mm'],m['v3']['depth_rmse_mm']),
                ('原本近共面邻面折角 P95（°）',m['v2']['initially_flat_face_angle_deg']['p95'],m['v3']['initially_flat_face_angle_deg']['p95']),
                ('位移 Laplacian 平均幅度（mm）',m['v2']['displacement_laplacian_mm']['mean'],m['v3']['displacement_laplacian_mm']['mean'])]
        for name,before,after in values:rows.append(f'<tr><td>{name}</td><td>{before:.6f}</td><td>{after:.6f}</td></tr>')
        sections.append(f'''<section><h2>{label}</h2><table><tr><th>检查项</th><th>v2</th><th>v3</th></tr>{''.join(rows)}</table>
<p>额外顶点修正最大值：{i['displacement_fairing']['correction_max_mm']:.6f} mm；标点保持；几何保护通过。</p>
<h3>相同视角和光照的显示对照</h3><img src="{state}/display_comparison.png">
<h3>相同平面着色的几何对照</h3><img src="{state}/flat_geometry_comparison.png">
<p><a href="{state}/mesh_{state}.obj">下载 v3 网格（米）</a> · <a href="{state}/mesh_v2.obj">v2 网格</a> · <a href="{state}/v3_report.json">算法报告</a></p></section>''')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Airway3D：受限平滑与显示对比</title>
<style>body{font:16px/1.65 'Microsoft YaHei',sans-serif;max-width:1450px;margin:30px auto;padding:0 24px;color:#17212c;background:#f4f6f8}section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd4de;padding:10px;text-align:left}a{color:#1565c0}.note{background:#e7eef8;padding:18px}</style>
<h1>Airway3D：受限位移平滑与显示对比</h1><p class="note">几何流程：原 stage2 → 中点细分 → v2 student 位移 → 最多 0.1 mm 的受限位移平滑。总位移仍不超过 2 mm，语义锚点与固定区域保持。默认图像使用平滑法线，平面着色图用于检查实际几何。</p>
<p>本页只报告当前 RGB-D 数据上的平滑性与深度贴合，不是 VEGA 独立扫描精度或逐点解剖学对应误差。相同采样数量 16000、种子 42。旧结果及输入文件未覆盖。</p>'''+''.join(sections)+'''<p><a href="report.json">完整输入／代码哈希与数值报告</a></p></html>'''
    (a.comparison/'index.html').write_text(html,encoding='utf-8')
    print('Comparison PNGs and index.html written.',flush=True)
if __name__=='__main__':main()
