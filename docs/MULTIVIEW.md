# 多视角 RGB-D（v0.9.0）

实际调用链：`MultiCaptureWidget → MultiPipelineWorker → run_multiview → fit_multiview → MultiProblem`。
原双帧 `CaptureWidget → PipelineWorker → run_pair` 路径保留。
联合细化调用链为`finish_multiview_refinement → refine_multiview_mesh → refine_state → refine_pointcloud`，只按状态调用两次，不按视角覆盖。

## 采集

运行 `start.cmd`，默认选择“闭口三视角 + 张口三视角”，依次采集闭口正面、左约40°、右约40°，再张口正面、左约40°、右约40°。
可切回四组（闭口三视角+张口正面）或原双帧模式。张口三次必须保持同样开口程度和头颈姿态。
张口侧面至少需要两组有效唇距，并检查配准后唇点位置；不足或不一致时提示重采。
角度只是引导。人保持头颈姿态及同状态表情，移动相机；每个位置停稳后点击采集。
每次融合该视角最近3–5帧有效深度中位数，记录相机线程采集时间、实际内参、深度抖动与质量权重。
切换视角或警告后清空短序列，禁止跨视角融合。正面闭口需ArUco，侧面不需要。
每帧独立保存内参/时间快照，丢弃切换视角或警告之前采集、之后才送达界面的排队帧。
不支持转头后把颈部当作随头刚性运动；声明为head_turned的数据拒绝。
运动过大、正面标点无效、稳定重叠不足、配准失败、开口程度变化或头颈不一致时提示重采。
自动检查不能证明人完全不动，仍需遵守采集操作。
`output/capture_mv_*/attempt_*/`保留每次原始尝试，`complete/session.json`是完整会话。
每次尝试的`raw_sequence.npz`含原始RGB/深度/内参/时间；`attempt.json`记录成功/失败及质量，停稳或检测失败也保留。
重建另存到`output/mv_*/`，不覆盖历史输入或结果。

## 离线回放与对照

在项目根目录的airway3d Python环境执行：

```powershell
python tools/replay_multiview.py --session output/capture_mv_时间戳/complete/session.json
python tools/replay_multiview.py --session 会话目录/session.json --compare-front
python tools/replay_multiview.py --legacy-pair output/20261027_143701_413530
```

`--front-only`只使用两种状态正面；`--compare-front`运行该基线和全部视角，分别保存。
第一阶段均关闭stage3，比较`closed/mesh_closed_stage2.obj`，保留张口正面网格及指标。
v0.9默认启用联合细化。`--stage2-only`或界面取消“联合细化网格”可关闭；`--compare-front`总是关闭stage3，保证第二阶段对照。
`--quick`减少迭代和样本，仅检查可执行性，不能作为收敛或精度证据。
`--output`指定新输出根目录。回放`prepared/session.json`可复用缓存分割。
原双帧仍可通过原界面模式及`run_pair`调用。

## 参数、坐标和诊断

`FitOptions`保留20/5→50/10→100/30渐进调度与LM停止条件。
所有观测共享shape，同状态共享expression和颈部/下颌屈伸；各视角独立SE(3)，无缩放。
坐标X右、Y上、前方Z负；深度毫米，点和网格米。
`extrinsics.json`为`p_view=R @ p_state_front+t`；两状态网格各在自己的正面相机系。
`registration_*_front.ply`蓝色为正面稳定点，橙色为变换到同状态正面的侧面稳定点。
配准用内部关键点RANSAC候选，再由排除嘴/下颌边缘/颈部的稳定皮肤多尺度ICP验证。
侧面检测标点须通过模型可见性、深度、重投影及三维一致性检查，不直接信任遮挡侧输出。
轮廓嵌入使用实际外部相机旋转与颈链乘积；LM内层冻结轮廓和对应，导数采用相同嵌入。
侧面只启用可靠深度和内部语义，不启用下颌轮廓；正面偏航超过20°记录回退。
新增多视角拟合不把任何0–16轮廓点（包括正面下巴8）作为三维锚点；下巴观测保留给原指标，正面下颌使用二维法向项。
视角数据项按有效权重归一后按质量加权；shape正则一次，expression每状态一次。

## 指标与验收

指标仍基于stage2。`report.json.metric_sources`指明两种状态正面及闭口ArUco来源。
网格、标点与ArUco在对应正面相机系，侧面标点不混入指标。原指标公式不变。
合成几何仅验证数学实现；已有真实正面数据仅验证离线可执行性。
缺少真实侧面及独立斜侧数据，**真实侧面改善尚未验证**。
真实验收应固定同组数据与stage2调度，比较仅正面和多视角；用未参与拟合的斜侧观测，分别报告正面/侧面表面距离、轮廓法向误差及有效覆盖率。
总损失下降、合成或正面复制数据不能代替真实验收。

## 联合顶点细化与配置

每状态只细分一次，只求解一个D。相机内的点目标通过`R.T @ (p_view-t)`回到同状态正面；三维各向同性残差在刚性旋转下等价。
语义方程`C_view*(R*V+t)=b_view`映射为`C_common=C_view*kron(I,R)`、`b_common=b_view-C_view*tile(t)`；导数使用相同映射。
在每个实际相机系执行深度置信度、投影、朝向、射线遮挡、唇部对应及语义可见性检查，然后把残差加入同一个稀疏系统。
可动区取各视角可见区及过渡/语义支撑的并集；每视角最多16000可靠深度点，各自归一化后按质量分配数据预算。
语义方程按各视角实际有效行数归一化，再按质量加权；统一预算取有效视角的平均方程数，单视角保持原有尺度。
`core/multiview_refine.py`设置平滑档4/1/0.25、每档8轮、Huber2mm、对应门限20mm、中点一致性16。
保留`RefineOptions/ShapeGuard`：总位移2mm、单次0.25mm、末端额外0.1mm、边长0.8–1.25、面积0.7–1.4、法线20°、折角10°。
不提供放宽这些保护的界面选项。末端只平滑D，不平滑基准坐标；冻结所有语义方程涉及顶点。输出`V=reference_vertices+offsets`一次。
`FitOptions`可配置LM档位、样本数、迭代及正则；会话`quality.weight`必须在(0,1]，`motion_mode`必须为`camera_moved`。
配准多尺度体素8/4/2mm、对应门限30/15/8mm，要求双向稳定区域重叠至少25%、RMSE不超过4mm；张口唇距变化超过max(3mm,15%)或配准后唇点中位偏差超过6mm拒绝。
射线分批128，降低内存峰值；无额外射线加速依赖，细分网格的严格可见性检查可能耗时数分钟。

已有stage2可以单独重放细化，不修改源目录或指标：

```powershell
python tools/refine_multiview.py --run output/mv_时间戳
```

新结果`output/mv_refine_*/`含两状态stage2副本、最终网格、细分基准和位移、保护报告；指标逐字节沿用原stage2结果。
正常完整重建另含`capture/`、`prepared/`、参数、外参、配准诊断、版本及输入/源码哈希。
可运行`python tools/check_multiview_outputs.py --run output/mv_时间戳`检查保存后的共享参数、外参、单次位移应用和全部保护；验证报告另存，不改源结果。

## 独立评估操作

```powershell
python tools/evaluate_multiview.py --run output/mv_时间戳 --geometry stage2
python tools/evaluate_multiview.py --run output/mv_时间戳 --geometry stage2 --held-out 独立会话/session.json --held-out-extrinsics 独立外参.json --reference-masks 人工核对掩码目录
```

独立观测必须有不同view_id且RGB/深度哈希不与拟合输入重复；不能用改名的同一采集代替独立视角。
独立外参需从标定获得，JSON格式与`extrinsics.json`相同，明确`direction=state_front_to_view`。评估不再拟合独立数据的相机/形状/D。
参考掩码为`view_id.png`，应是与完整模型外轮廓范围一致、人工核对的二值轮廓；未提供时轮廓误差为缺项，不能把检测器标点当轮廓真值。
报告按`in_fit/held_out`及`front/side`分开，包含未按距离裁剪的点到可见三角面的均值/RMSE/P95、轮廓法向像素误差与双向Chamfer、可靠深度比例及20mm门限下有效覆盖率。
表面误差使用可靠深度（仍有传感器噪声），不能据此声称解剖真值。可改`--geometry stage3`评估最终几何；先完成同组stage2对照。
没有独立会话时仅给拟合内诊断，`independent_views_present=false`；没有参考掩码时不填写轮廓数值。
