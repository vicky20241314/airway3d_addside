# v0.13.0 算法验证记录（2026-10-30）

本轮完成重建代码、73项自动测试、两类真实会话各11个正式拟合配置、共享形状CPU重求及修正网格联合细化。README和CHANGELOG已更新。未改动独立评估目录或评估定义。

## 结果性质与缺项

以下全部是拟合输入上的共同观测诊断：所有模型使用C的同一初始ICP相机、每视图4000点预算和种子42，沿用原evaluate_view/depth_inputs。其可靠样本仍可能依赖模型支持；不是独立标定，也不是固定样本验证。未提供独立斜侧、可信头动外参或人工核对轮廓，轮廓误差记为缺失，不填零。不能据此宣称真实侧脸精度验收通过。

原始输入、旧输出、旧算法可选路径和几何保护保留；共享低频默认关闭。最终默认D：温和prior、旧关键点筛选、配准回退；joint保留，soft/fixed/轮廓/低频为显式候选。J为joint+soft候选，不是最终默认。相同调度20/5→50/10→100/30，点预算4000、拟合种子1234。旧兼容路径保留原损失/策略，但ICP统一修复了完全重合时零距离点被全部删除的问题，不声明位级重放一致；没有放宽原数值回归门限。

## 软件与保护

最终默认配置全量 unittest：73/73，退出0；耗时见tests_final.log。覆盖坐标/投影、共享参数、观测掩码、遮挡、SE(3)方向、三个策略的残差雅可比、fixed渐进active映射、轮廓雅可比、ROI/SIFT/三维几何种子成功与退化、无重叠诊断保存、缺唇降级、低频刚体/关节零矩、跨姿态保护、保存重建和D只加一次。

阶段3保持2 mm总位移、0.25 mm单步、0.1 mm额外平滑，以及边长0.8–1.25、面积0.7–1.4、法线20°、折角10°保护；低频修正单独最多3 mm，所有姿态检查保护区域1微米数值容差，未与stage3位移混算。业务metrics仍为原stage2。

## 正式消融

表中为点数加权RMSE（mm）；逐视图mean/RMSE/P95、旧工具定义的有效覆盖率、等视图RMSE、实际视角、配置与来源见配套JSON。低频继承运行的耗时仅为alpha重求，原拟合耗时单列于JSON；这些运行存在其他检查任务负载，不能作为速度基准。

### good：capture_mv_20261030_095649_134226

缺少共同参考相机的视角：无。未评分视角不记为零误差。

| 配置 | 几何 | 正面RMSE | 侧面RMSE | 拟合视角数 |
|---|---|---:|---:|---:|
| A_front_stage2 | stage2 | 2.006 | 2.613 | 2 |
| B_legacy_multiview | stage2 | 1.926 | 2.432 | 4 |
| C_registration | stage2 | 1.926 | 2.432 | 4 |
| D_pose_prior | stage2 | 1.827 | 2.001 | 4 |
| E_soft_landmarks | stage2 | 1.738 | 1.961 | 4 |
| F_profile | stage2 | 1.738 | 1.961 | 4 |
| G_canonical | shape_corrected | 1.715 | 1.930 | 4 |
| H_profile_stage3 | stage3 | 1.222 | 1.328 | 4 |
| G_front_capacity | shape_corrected | 1.991 | 2.598 | 2 |
| A_front_stage3 | stage3 | 1.406 | 2.087 | 2 |
| G_stage2_parent | stage2 | 1.738 | 1.961 | 4 |
| G_front_stage2_parent | stage2 | 2.006 | 2.613 | 2 |
| J_joint_soft_candidate | stage2 | 1.925 | 2.390 | 4 |

F最终可靠侧面轮廓方程：0。0表示没有可靠片段，本项未生效；不能将F/E数值差解释为轮廓收益。

### challenging：capture_mv_20261030_120114_381948

缺少共同参考相机的视角：['closed_left']。未评分视角不记为零误差。

| 配置 | 几何 | 正面RMSE | 侧面RMSE | 拟合视角数 |
|---|---|---:|---:|---:|
| A_front_stage2 | stage2 | 2.590 | 3.549 | 2 |
| B_legacy_multiview | stage2 | 2.589 | 3.550 | 2 |
| C_registration | stage2 | 2.680 | 3.650 | 3 |
| D_pose_prior | stage2 | 2.670 | 3.021 | 3 |
| E_soft_landmarks | stage2 | 2.873 | 2.089 | 3 |
| F_profile | stage2 | 2.874 | 2.088 | 3 |
| G_canonical | shape_corrected | 2.844 | 2.074 | 3 |
| H_profile_stage3 | stage3 | 2.420 | 1.901 | 3 |
| G_front_capacity | shape_corrected | 2.574 | 3.539 | 2 |
| A_front_stage3 | stage3 | 2.087 | 3.066 | 2 |
| G_stage2_parent | stage2 | 2.873 | 2.090 | 3 |
| G_front_stage2_parent | stage2 | 2.590 | 3.552 | 2 |
| J_joint_soft_candidate | stage2 | 2.653 | 3.662 | 3 |

F最终可靠侧面轮廓方程：0。0表示没有可靠片段，本项未生效；不能将F/E数值差解释为轮廓收益。

## 判断与剩余验收

默认选D而非E/J：good的B→D侧面2.432→2.001 mm，正面1.926→1.827 mm；challenging的B→D侧面3.550→3.021 mm，正面2.589→2.670 mm。C→D在相同有效视角下也有诊断改善。但共同相机来自初始ICP，先验也锚定该ICP，不能把差值视为独立形状精度证据。soft的E在低质量样本正面2.670→2.873 mm，故不默认开启；J_joint_soft_candidate保留该取舍证据。

低频G与其自己的stage2_parent比较，另有front-only同容量对照；修正幅度较小，不能据此断言FLAME容量不足已被解决。修正后的全量关节零矩与保护区域检查通过，仍默认关闭。F可靠片段不足时主动不加项；没有用放松深度孔洞或语义条件来制造轮廓收益。

good的E/F/G父stage2在轮廓无方程时构成同一有效目标的重复完整拟合，侧面RMSE变化约0.0004 mm；配置文件并非逐字相同，不把它替代最终独立测试。没有重跑所有历史双帧/六视角或新硬件采集。两会话不支持跨人泛化、未观测后脑厚度或张口侧面效果结论。

第二类存在被排除侧面，原始数据保留且其余视角完成；共同相机缺失的侧面没有评分。三维特征回退的自动成功/退化测试通过，不能据本轮全部可用视角就宣称真实配准成功率提高。

真实验收缺项：独立且未调参的斜侧、与模型无关的头部参考外参、状态一致性、人工轮廓，以及固定观测样本支持率评估。按本轮优先重建的要求，评估工具改造延期；没有用总拟合损失当作效果验收。

并行GPU附加细化曾显存不足（退出1）；改为串行后修正网格细化及输出检查均退出0，闭口/张口最大位移1.225/1.180 mm，指标字节相同。首次轮廓接线试跑和所有失败/中断目录均保留，不计作完整发布结果。

## 复现与产物

在项目目录使用 `D:\softwares\miniconda\envs\airway3d\python.exe`。完整操作及参数见 [算法说明](ALGORITHM_V013.md)，消融入口 `tools/replay_multiview.py --experiments configs/multiview_ablation.json`；输出检查复用 `tools/check_multiview_outputs.py`。

实际运行根目录：`output/algorithm_v013_20261030_good/`、`output/algorithm_v013_20261030_challenging/`。保留replay_runs.json、所有日志、ablation_summary.json/md、固定相机图、逐视角叠加、0–6 mm几何变化PLY及源数据摘要。本轮摘要/绘图复用函数的调用脚本保留在 `output/algorithm_v013_summarize.py`；`--resume-summary --no-render`只更新新结果，`--render-only`重绘，未改原评估实现。

配套 [机器可读结果](ALGORITHM_V013_RESULTS.json) 包含逐视图、状态与聚合统计、位姿变化、源哈希和当前25个成功输出检查来源；默认D的两套stage3另行回放并保持指标字节。采集和大图遵守原忽略规则，不推入Git。
