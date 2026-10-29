# 多视角 RGB-D（v0.11.1）

实际调用链：`MultiCaptureWidget → MultiPipelineWorker → run_multiview → fit_multiview → MultiProblem`。
原双帧 `CaptureWidget → PipelineWorker → run_pair` 路径保留。
联合细化调用链为`finish_multiview_refinement → refine_multiview_mesh → refine_state → refine_pointcloud`，只按状态调用两次，不按视角覆盖。

## 采集

运行 `start.cmd`，默认选择“闭口三视角 + 张口正面（推荐）”，依次采集闭口正面、左约40°、右约40°，最后仅张口正面。闭口侧面约束共享脸型，张口正面约束张口状态和指标，无需长时间跨视角保持张口。
可选择额外采集张口侧面，或原双帧模式。额外张口侧面是可选观测；缺少可靠可见唇点、张口程度不同、配准或一致性不通过时排除整个视角，保留其原始数据，继续使用张口正面。没有伪造缺失唇点，也没有将不同张口强行共用同一状态。
只有通过配准与表情一致性检查的张口侧面才加入共享状态拟合；没有增加每视角表情自由度。结果页显示实际采用数量和未采用的张口视角；`view_selection.json`、`report.json.excluded_views`保留原因。闭口侧面配准失败仍需重采。
角度只是引导。人保持头颈姿态及同状态表情，移动相机；每个位置停稳后点击采集。
默认使用该视角最近5帧，可选9/15帧。相机到位后短暂停留，等界面显示已缓存5/5后点击采集（30fps时约0.17秒）。只使用1秒以内、同内参、同状态/视角的短序列。
逐帧检测关键点，剔除相对模糊、无法补偿的运动或明显张口变化。至少保留3帧且不少于输入帧数的60%；5/9/15帧分别至少保留3/6/9帧。帧数不是越多越好，建议先用5帧。
稳定区原始位移不超过2像素/3mm时直接使用。超出时先通过上半脸RGB双向追踪和深度估计帧→参考帧的无尺度刚性变换，再检查未参与估计的追踪点。限3°、表面点中位移动15mm；独立追踪点中位残差不超过2mm且至少70%在3mm内，补偿后语义点中位偏差不超过3mm，否则剔除。记录每帧外参和检查结果。
补偿后的深度使用Z-buffer投影到参考相机，要求与参考帧有效深度相差不超过5mm，不填补参考深度孔洞、不混合遮挡前后表面；关键点也检查参考深度。唇距变化门限仍为max(2mm,10%)，不使用嘴部拟合相机运动。头颈应保持自然姿态；小幅头动与相机运动不能仅靠上半脸严格区分，自动检查不是姿态不变的证明。
深度至少有3帧且不少于一半有效支持，时间MAD不超过2mm；关键点至少3帧且60%有效票，二维离散度不超过2像素、三维不超过4mm。最终三维点由共识二维坐标和多帧关键点深度计算，保持投影一致。
以同一参考帧RGB用于分割/显示，深度和关键点在其相机系融合；几何约束不再仅来自该帧。时间一致性不能证明侧面遮挡点可靠，后续可见性检查仍执行。
切换视角或警告后清空短序列，禁止跨视角融合。正面闭口需ArUco，侧面不需要。
每帧独立保存内参/时间快照，丢弃切换视角或警告之前采集、之后才送达界面的排队帧。
不支持转头后把颈部当作随头刚性运动；声明为head_turned的数据拒绝。
运动过大、正面标点无效或闭口侧面配准失败时提示重采；可选张口侧面不可靠则排除，不阻断其余观测。
自动检查不能证明人完全不动，仍需遵守采集操作。
`output/capture_mv_*/attempt_*/`保留每次原始尝试，`complete/session.json`是完整会话。
每次尝试的`raw_sequence.npz`含原始RGB/深度/内参/时间；`attempt.json`记录成功/失败及质量，停稳或检测失败也保留。
会话观测文件可选包含`temporal_*`数组：原始与补偿后二维/三维关键点、各自有效标记、帧→参考帧外参、帧接受标记、时间、逐像素深度支持次数与MAD；旧会话仍可读取。
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
快速结果附带`QUICK_TEST_ONLY.txt`，`report.json.run_purpose=smoke_test`；所有新多视角运行保存`fit_configuration`。正式回放去掉`--quick`，才使用4000点、10轮外参优化和每档最多20轮联合LM（可提前收敛）。
`--output`指定新输出根目录。回放`prepared/session.json`可复用缓存分割。
原双帧仍可通过原界面模式及`run_pair`调用。

保留了原始采集短序列时，可离线重新进行多帧筛选和融合，输出新会话：

```powershell
python tools/prepare_multiframe_capture.py --capture output/capture_mv_时间戳
python tools/replay_multiview.py --session output/temporal_capture_时间戳/complete/session.json
```

只有融合后的观测时，不能恢复真实逐帧信息；不能把复制同一张RGB-D当作新的多帧采集。

## 参数、坐标和诊断

`FitOptions`保留20/5→50/10→100/30渐进调度与LM停止条件。
CLI可用`--shape-reg 0.0000002`对身份先验做单参数试验，默认仍为`0.000001`，要求有限正数。减弱先验会增加个体形状自由度，也可能拟合噪声；必须记录配置并做同组/独立数据对照，不能凭单组损失下降改变默认。
所有观测共享shape，同状态共享expression和颈部/下颌屈伸；各视角独立SE(3)，无缩放。
坐标X右、Y上、前方Z负；深度毫米，点和网格米。
`extrinsics.json`为`p_view=R @ p_state_front+t`；两状态网格各在自己的正面相机系。
`registration_*_front.ply`蓝色为正面稳定点，橙色为变换到同状态正面的侧面稳定点。
配准用内部关键点RANSAC候选，再由排除嘴/下颌边缘/颈部的稳定皮肤多尺度ICP验证。
v0.9.1颈部检查只比较双向投影到共同颈部语义区域的可靠表面，使用点到面误差，保留10mm拒绝阈值；不按误差大小提前删点。分割边界、孔洞/跳变与不相容朝向不参与比较。
`registration_视角.json.neck_check.status`区分`passed`、`partial_overlap`、`insufficient_overlap`和`inconsistent`。不足重叠只表示头颈一致性未完整验证，会输出警告；不能据此证明采集姿态正确。
逐视角JSON保留side→front外参、ICP和颈部统计。表情或颈部一致性失败时仍保存`registration_视角_failed.json`及`registration_视角_failed_front.ply`，蓝色/橙色含义相同。
侧面检测标点须通过模型可见性、深度、重投影及三维一致性检查，不直接信任遮挡侧输出。
轮廓嵌入使用实际外部相机旋转与颈链乘积；LM内层冻结轮廓和对应，导数采用相同嵌入。
侧面只启用可靠深度和内部语义，不启用下颌轮廓；正面偏航超过20°记录回退。
新增多视角拟合不把任何0–16轮廓点（包括正面下巴8）作为三维锚点；下巴观测保留给原指标，正面下颌使用二维法向项。
视角数据项按有效权重归一后按质量加权；shape正则一次，expression每状态一次。

## 指标与验收

指标仍基于stage2。`report.json.metric_sources`指明两种状态正面及闭口ArUco来源。
网格、标点与ArUco在对应正面相机系，侧面标点不混入指标。原指标公式不变。
合成几何仅验证数学实现；已对真实手持5帧原始序列进行重处理和可用性检查，不能代替独立精度验收。此前已有真实六视角完整调度对照，尚未证明相似度问题解决。
缺少独立斜侧验收数据和人工核对轮廓，**真实侧面精度改善尚未验证**。
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
默认通过64×64投影空间网格筛选射线候选三角面，再执行精确三维交点和重心包含检查；网格仅用于候选索引，不是渲染分辨率，不对射线/网格做降采样。候选按预算分批处理。跨相机平面或异常投影范围回退原分批128条射线的实现，恰好位于距离容差边界的点也由原实现复核，无新增依赖。

已有stage2可以单独重放细化，不修改源目录或指标：

```powershell
python tools/refine_multiview.py --run output/mv_时间戳
```

新结果`output/mv_refine_*/`含两状态stage2副本、最终网格、细分基准和位移、保护报告；指标逐字节沿用原stage2结果。
正常完整重建另含`capture/`、`prepared/`、参数、外参、配准诊断、版本及输入/源码哈希。
`capture/`和`prepared/`保留采集观测；`report.json.captured_view_ids`与`view_ids`区分输入和实际拟合视角。细化仅使用已拟合外参的视角；评估工具明确记录未参与拟合的输入，不将其冒充独立留出数据。
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

## 配准失败后的缓存诊断

失败目录的`prepared/session.json`已包含分割和标点，可先运行快速配准检查，无需重采或先运行FLAME拟合：

```powershell
python tools/check_multiview_registration.py --session output/mv_时间戳/prepared/session.json
python tools/replay_multiview.py --session output/mv_时间戳/prepared/session.json --stage2-only
```

快速检查另存`output/registration_check_*/`，只验证初始配准及一致性，不代表最终重建精度。完整回放也写新目录，不覆盖失败时的采集数据。
逐次检查保留的原始采集尝试可运行`python tools/check_capture_bursts.py --capture output/capture_mv_时间戳`，输出旧/新帧筛选结果、运动补偿记录、正面完整性检查和输入哈希。检查工具退出0表示检查完成，仍须查看各次尝试是否通过。
表情变化或共同区域的颈部误差超限仍需重采；重叠不足时应人工检查叠加及头颈姿态，必要时补采更完整的正面/侧面颈部。

## 相同观察条件下的结果对照

```powershell
python tools/compare_multiview_runs.py --reference-run output/六视角完整运行 --run "Front only" output/仅正面运行 stage2 --run "Six views" output/六视角完整运行 stage2
```

工具以参考运行的初始面部ICP外参为共同相机，对所有结果使用相同内参、观察区域和光照，生成两状态对照PNG与`comparison.json`。可增加`--run "Refined" output/细化运行 stage3`。
表面误差不按距离裁剪，报告正面/侧面及有效覆盖率；逐视角标记是否用于该结果的几何拟合。相机外参来自这批观测本身，因此这些仍是共享观测诊断，不是独立精度评估；没有人工核对轮廓时不报告轮廓误差。
照片包含肤色、眉毛、头发与光照，而当前输出是无纹理几何。应同时检查几何对齐、鼻唇与下颌轮廓和深度误差，不能仅凭裸网格的整体观感或总损失判断精度。

## 性能诊断与回归

v0.6.0主要运行闭口/张口两个正面；当前多视角入口先完整执行正面初始化，再做联合拟合和各视角遮挡/细化，工作量更大。v0.11.1优化最慢的遮挡查询，保持原完整初始化和LM配置。
进度现在区分“正面初始化”“多视角相机对齐”“多视角联合拟合”，各状态细化结束显示耗时。`report.json.timing`记录各大阶段，`fitting.timing`细分配准/初始化/相机/联合拟合；细化的`total_elapsed_s`包含其准备工作，原`elapsed_s`仍表示求解器时间。

```powershell
python tools/benchmark_visibility.py --run output/已完成运行
python tools/check_speed_regression.py --baseline output/加速前运行 --run output/加速后运行
python tools/compare_refinement_backends.py --run output/已完成运行
```

第一个工具逐视角比较旧/新射线判断及计时；第二个要求相同原始输入、内参、拟合配置和实际视角，再比较网格与指标。数值回归门限为stage2最大顶点差0.005mm、stage3最大差0.02mm，不代表真实解剖精度。
第三个工具固定保存的stage2、外参及观测，在同一进程分别运行两种遮挡后端，隔离完整重复拟合的差异；原后端较慢，可能需要十余分钟。v0.11.1已完成这项受控对照，两状态最终顶点差为0。完整重复拟合曾超出上述严格回归门限，报告保留在验证记录中，不能把受控细化一致误写成从头重建完全一致。
如需复核旧遮挡实现，可在当前终端设置`$env:AIRWAY3D_RAY_BACKEND='reference'`再回放；恢复默认用`Remove-Item Env:AIRWAY3D_RAY_BACKEND`。该设置只改变查询实现，不改变采样与迭代，运行报告记录所选后端。不同机器/数据/并发负载的耗时不同，局部查询倍数不能当作整段重建倍数。
