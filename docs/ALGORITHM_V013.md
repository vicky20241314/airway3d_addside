# v0.13 重建算法与复现

本轮优先修改重建；`reconstruction_iteration_eval/`、现有评估样本定义和 held-out 工具未修改。独立真实侧脸精度未验收。逐次运行结果见 [验证记录](ALGORITHM_V013_VALIDATION.md)。

## 实际调用链与默认行为

`MultiCaptureWidget → MultiPipelineWorker → run_multiview → select_registered_views → train(正面初始化) → MultiProblem → solve_stage(相机) → solve_progressive(共享形状/状态) → 可选 correct_shape → stage2 指标 → finish_multiview_refinement`。

原双帧 `run_pair`、原始采集和旧输出保留。共享一套 shape；同状态共享 expression/关节；各视角独立米制 SE(3)，无自由尺度。正面 ArUco 和标点始终是业务指标来源；`metrics.json` 仍为原 stage2 几何，实验修正和 stage3 不回写。

- 默认 `registration_fallback=true`。有有效标点时优先原种子；失效时，分割中的眼眉和鼻必须支持保守上半脸 ROI，腐蚀语义边界，剔除深度孔洞/跳变及侧向法线。无可靠标点种子时，用双向 ratio-test SIFT 特征反投影到实际 RGB-D，再进行三维刚性 RANSAC。SIFT 不足时尝试点对法线局部坐标的三组11-bin几何直方图（SPFH形式）互检匹配；该几何候选对低曲率/重复表面仍可能退化，两种特征都不可靠时明确失败，不用单位变换盲跑 ICP。所有候选经法线相容多尺度 ICP、8 mm 双向重叠、25% 最低双向支持、4 mm RMSE 与空间退化检查。几何特征回退尚需更多真实失败样本验证，不能保证补救全部侧面。
- 默认 `side_expression_policy="upper_face"`。缺唇为 `unverified`；唇距/位置不一致为 `inconsistent`。这两种情况均禁用下脸，只保留经过独立检查的上半脸，状态共享仍不变。若目标是张口下颌，应重新采集更一致的张口斜侧；当前降级不能补出未观测的张口下颌。`strict` 保留旧行为。
- 默认 `side_landmark_mode="legacy"` 保留旧筛选。显式 `soft` 实验只选 `21,22,27..47` 静态内部点；要求 detector valid、有限真实深度、可靠语义、局部深度连续和模型可见。用三维残差 Huber IRLS（4 mm）连续降权，20 mm 为明显粗差上限。外层刷新，内层固定；不把侧面外轮廓/唇部当作固定三维材料点。诊断记录残差、权重及旧 6 mm 规则会删除的点数。低质量样本的soft正面退化，因此不默认开启。
- 默认 `pose_strategy="prior"`，采用下述温和相对先验，对应消融D；两套数据未支持默认同时打开soft。`joint` 保留自由联合优化，`fixed` 仅为高可信/受控消融，不宣称任何策略普遍更好。

## 相对位姿定义

`p_view=R_view p_FLAME+t_view`；`T_rel=T_side inverse(T_front)`，即状态正面相机坐标 → 侧相机坐标。初始 `T_rel` 来自侧面到正面配准的逆。先验使用左差 `Log(T_rel inverse(T_rel_initial))`，完整 SE(3) 对数的平移分量包含左雅可比逆。旋转标准差6°、平移15 mm，只是显式保守配置，不将ICP RMSE当协方差。白化先验平方权重 `1e-6`，与米制数据项分开记录。

`fixed` 在相机阶段后把同状态相对外参绑定到初始配准，仅移除侧面的6个独立自由度；每状态正面的全局对齐仍可优化，并通过相对变换传到侧面。基底调度及 active 参数映射一致，数值测试覆盖实际残差导数。

## 轮廓候选（默认关闭）

`side_contour_weight=0.3` 可启用候选。从头面部分割的外轮廓提取朝向鼻部一侧的片段，要求眼眉/鼻锚、4像素内侧可靠深度，排除耳/头发/颈部/衣领/图像边界附近片段；缺唇一致性时只留上半脸片段。没有可信片段就明确报告，不盲目使用完整 mask 外边界。

模型轮廓由当前相机中朝向改变的网格边及精确可见性构造；使用片段双向最近邻法向残差，每外层刷新，LM 内层固定轮廓边/目标/IRLS。像素噪声2 px、Huber3 px，白化后按1 mm数据预算与视角质量/方程数归一化。该噪声预算尚未独立标定；不能把轮廓约束开启等同于轮廓精度已改善。保存RGB叠加供人工检查。

## 低频共享修正（默认关闭）

`canonical_shape_enabled=true`：固定已拟合身份/状态/外参，只求解共享 `alpha`。可靠可见语义区域确定支持集，无观测、颈部、唇部保护区域零位移；Dirichlet图Laplacian低频基底去除6个刚体场，默认12个标量模式，最多36个向量系数。`V_rest=V_template+B alpha`，经原FLAME表达式与LBS，关节从修正后 shaped rest vertices 回归；所有状态共享同一修正。

总修正上限3 mm（配置可收紧），含零位移及低频平滑先验。基底还约束所有关节回归量为零，避免即使颈部静止位移为零仍因关节枢轴变化而间接移动；继续走原关节回归/LBS，检查姿态后保护区域最大变化不超过1微米数值容差。静止空间和所有姿态/相机输出均检查边长、面积、法线、折角保护，并以真实FLAME重新计算核验线性导数输出。这个模块补充受限模型容量，不代表侧面观测带来的收益，必须对照 front-only 的同容量版本。

输出 `canonical_shape.npz`（alpha、basis、active、offset、拓扑和基底哈希）、`canonical_shape_report.json`、`closed/open/mesh_*_shape_corrected.obj`。不冒充原stage2。默认细化仍从 `stage2` 输入；如设置 `refinement_source="shape_corrected"`，必须同时启用低频修正，保存来源及相对原stage2累计变化。stage3 2 mm/0.25 mm/0.1 mm保护及位移只加一次的合同不变。

## 采集与回放

启动 `start.cmd`，选择四组模式。每个方向采5帧短序列，至少3帧通过已有手持补偿和质量检查；不用为了凑15帧长时间憋住表情。闭口正面需看到 ArUco；左右斜侧先确保眼鼻上半脸重叠，角度约40°只是引导。移动相机或自然小幅头动均可；侧面颈部从不刚性并入头模。张口侧面可选，状态不一致只支持上半脸时会明确告知。大幅运动、模糊、孔洞或无可信配准仍会拒绝对应视角，其余可用视角继续；两张正面仍严格。

在 `D:\proiect\newstage2_oldstage3` 的 PowerShell 中运行（换用新的输出目录）：

```powershell
& 'D:\softwares\miniconda\envs\airway3d\python.exe' -X utf8 tools\replay_multiview.py --session 'output\capture_mv_20261030_095649_134226\complete\session.json' --experiments configs\multiview_ablation.json --output output\my_algorithm_ablation
& 'D:\softwares\miniconda\envs\airway3d\python.exe' -X utf8 tools\replay_multiview.py --session '<session.json>' --pose-strategy prior --stage2-only --output output\my_prior
```

`--config <JSON>` 覆盖 `FitOptions`，未知字段报错；`--experiments` 为按顺序执行的配置列表，支持 `label/options/front_only/refine`，每完成一项更新 `replay_runs.json`。批量与 `--compare-front` 复用同一哈希核验的 `front_initialization.npz`；正面输入/原拟合配置变化会明确拒绝缓存。每项独立保存完整原始及预处理快照、配置、源码哈希和初始化签名。不要复用其他数据的输出根目录。`--quick` 仅用于流程检查，不能用于质量判断。

中断后用相同命令加 `--resume`，逐项核验原始观测和已保存配置后跳过完整结果；失败/未完成目录保留。`--geometry-maps` 额外保存去除相机刚性对齐的几何变化PLY，固定0–6 mm色标；它显示形状/状态变化，不是精度误差热图。已保存的实验修正可直接用 `tools/refine_multiview.py --run <G目录> --geometry shape_corrected --output <新目录>` 细化，不必重拟合；该入口复制原stage2和指标字节，另存细化结果。

仅更新低频模块时，用 `tools/replay_multiview.py --fitted-run <完整stage2结果目录> --output <新目录>` 在CPU上重求alpha；继承身份/状态/相机，不重拟合，原stage2及指标按字节复制。输出记录原运行、继承配置、源码摘要及本次alpha求解耗时。该模式不接受其他拟合配置覆盖，防止将改变的初始化混入容量对照。

`configs/multiview_ablation.json` 包含 A前向、B旧多视角、C配准、D位姿先验、E软点、F轮廓、G低频、H细化，以及G前向同容量与A前向同算法细化。默认不修改迭代/点数/shape正则。复用 `tools/compare_multiview_runs.py` 进行相同初始ICP相机的对照，复用 `tools/evaluate_multiview.py --held-out ...` 做有独立外参的留出验证；本轮不变更这些工具的评估定义。

## 诊断与边界

每次保存 `view_selection.json`、`fitting_report.json` 的逐视角参与/排除/对应数/鲁棒权重/各残差平方和、各外层相对位姿漂移，`solver_diagnostics.md` 和 `stage2_overlay_*.png`。报告明确全部侧面被排除后的正面降级。点数与动态对应 RMSE 不能当作覆盖率或固定样本精度。现有评估对模型支持有依赖，固定ICP坐标也不是独立标定，相关数字只能作拟合内诊断。

独立验收仍需未参与拟合和调参的不同斜侧角度、可信头部参考外参、状态一致性和人工核对轮廓。跨时刻相机标定不能替代头动补偿。同角度burst只降噪，不是独立留出。缺少这些数据时不声明侧面/下颌角/厚度精度通过，也不从头部网格推断气道精度。
