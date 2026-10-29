# v3：受限位移平滑与显示改进（2026-10-25）

## 当前采用的方案

版本为 `cross-v3-bounded-smooth`，默认通过采集界面调用。新版初始化、stage1/stage2、v2 的 student 求解、细分锚点和几何保护保留。

在 student 已接受位移 D 的基础上求小修正 X：

```text
min ||X||² + ||L(D+X)||²
```

其中 L 为原有位移 Laplacian，冻结语义方程涉及的全部顶点，并保持其他不可动顶点。每点额外修正不超过 0.1 mm，相对细分基准的总位移仍不超过 2 mm；几何检查和目标下降同时满足才接受，否则保留原结果。平滑作用于位移，未对 stage2 绝对坐标作整体平滑，不重新拟合 FLAME。

总位移统一保存为 `offsets = V_final - V_subdivided`。不把两份位移重复叠加。由于三角形拓扑相同，任一固定重心坐标表面点的移动也受 0.1 mm 上限约束；这是相对 v2 的变化界限，不是相对真实人脸的误差上限。

显示改为顶点法线平滑着色，保留 `smooth=False` 平面着色选项。真实 pyrender 已测试两种方式，顶点与面片输入保持不变。着色本身不会改善 OBJ 几何。

## 数据与公平对照

此前截图对应 `output/20261025_150014_977582/` 的采集文件在本轮执行时已不在原位置，因此没有声称重新评价了那次采集。

本轮使用仍在磁盘上的完整成对数据：

`validation/cross_v2_full/runs/20261025_153220_481039/`

该目录是此前完整离线回归保存的结果。v2、v3 使用同一份保存的 stage2、闭口／张口各自深度、分割、标点和相机内参。每帧固定抽取 16,000 个可靠深度点，seed=42。几何统计区域由初始可观测顶点决定，在比较前固定，不按结果修改。输入与源码哈希位于 `validation/cross_v3_smooth/20261025_153220_481039/report.json`。

## 真实 RGB-D 双帧结果

| 指标 | 闭口 v2 | 闭口 v3 | 张口 v2 | 张口 v3 |
|---|---:|---:|---:|---:|
| 原本近共面的邻面折角 P95（°） | 9.1871 | 5.2350 | 9.4687 | 5.2825 |
| 可观测区域位移 Laplacian 平均幅度（mm） | 0.05393 | 0.03584 | 0.05891 | 0.03738 |
| 固定深度点到三角形表面 RMSE（mm） | 1.75054 | 1.75702 | 1.55792 | 1.56204 |
| 51 点二维投影残差平均值（像素） | 5.69610 | 5.69610 | 5.65496 | 5.65496 |

近共面邻面折角 P95 分别下降约 43%／44%；这是一项局部平滑性统计，不是全脸误差降低百分比。深度 RMSE 增加约 0.00648／0.00412 mm，属于此次平滑的贴合代价。该结果支持“起伏减少且变化幅度小”，不支持宣称真实扫描精度提高。

双帧验证退出码 0，检查通过：

- stage2、RGB-D 和预处理输入不变；两种状态分别处理。
- 语义锚点位置相对 v2 保持（1e-12 m 数值容差）。
- 每点额外移动不超过 0.1 mm，总位移不超过 2 mm。
- 边长、面积、法线和邻面折角保护通过。
- 位移保存重组的最大舍入差：闭口 2.17e-19 m、张口 5.42e-20 m。
- 真实 pyrender 的平滑／平面着色均通过，输入几何没有被渲染过程修改。

原验证脚本曾使用逐位 `array_equal` 检查浮点减加重组，导致中断；已改为记录实际差值并要求小于 1e-12 m，未放宽任何几何保护条件。

## 合成验证

已知曲面包含保留的基准凸起和一个平滑的小幅真实位移，固定 4,000 个采样点，深度噪声标准差 0.4 mm，seed=20261025。真实顶点 RMSE 从 v2 的 0.08508 mm 降至 v3 的 0.08318 mm；几何约束、有效非零位移及人工尖峰识别通过。此单组合成结果不能代表跨人的真实精度。

结果：`validation/cross_v3_smooth/synthetic.json`。

## 未采用的候选

曾测试更强位移正则、切向约束和更长优化：闭口深度 RMSE 略降，但局部折角增加；张口虽较平滑，深度贴合及标点投影有所退化。合成真值 RMSE 从 0.08508 mm 变为 0.09381 mm。收益不一致，因此没有作为默认算法。

该实验的结果在 `validation/cross_v3/`，源码快照在 `docs/archive/rejected_more_fit_candidate/`。**当前采用的是 `cross_v3_smooth/` 中的受限平滑方案。** v2 原代码快照在 `docs/archive/cross_v2_before_v3/`。

## 查看与复现

- 对比页面：`validation/cross_v3_smooth/20261025_153220_481039/index.html`。
- closed／open 下的 `mesh_*.obj`：本轮 v3 网格，单位米。
- `display_comparison.png`：v2 平面着色、v2 仅改着色、v3 平滑着色。
- `flat_geometry_comparison.png`：同光照平面着色的几何对比。
- `vertex_offsets.npz`：细分基准、面片、最终总位移。
- `v3_report.json`：student 迭代、几何检查、平滑强度及实际修正量。

```powershell
python tools/check_v3_contracts.py
python tools/validate_cross_v3.py --run validation/cross_v2_full/runs/20261025_153220_481039 --render
python tools/check_stage3_pipeline.py --run validation/cross_v2_full/runs/20261025_153220_481039
python tools/build_v3_review.py --comparison validation/cross_v3_smooth/20261025_153220_481039
```

实际 `finish_meshes` 保存链路检查写入 `validation/cross_v3_smooth/pipeline/checks.json`。它使用真实指标与双帧 stage3，确认默认算法、stage2 隔离及各自输出与独立验证一致；没有把离线脚本当作采集程序入口。

本轮没有重新执行未修改的 stage1/stage2 拟合，也没有重新验证相机硬件。更高真实精度仍需独立扫描评价和多次采集验证；剩余的大尺度偏差不能通过视觉平滑解决。当前两份来源项目保持不变。
