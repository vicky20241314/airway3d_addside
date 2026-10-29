# Airway3D 交叉人脸重建版

**v0.8.0 多视角入口已加入。** 默认采集闭口和张口各正面/左右约40°，保持人不动、移动相机；界面可切回四组或原双帧模式。
多视角共享身份与同状态表情/姿态，逐视角外参，暂关闭第三阶段。采集、回放和限制见 [多视角说明](docs/MULTIVIEW.md)，验证见 [验证记录](docs/MULTIVIEW_VALIDATION.md)。
以下原有交叉版流程描述对应保留的双帧模式。

当前基线版本：**v0.6.0**（见 `VERSION`）。项目更新同步至 [GitHub 仓库](https://github.com/vicky20241314/airway3d_addside)。模型和大型二进制文件使用 Git LFS，克隆前请安装 Git LFS；已有克隆可执行 `git lfs pull` 获取这些文件。此版本记录现有算法基线，尚未实现多视角采集与联合拟合。

根目录 `start.cmd` 或 `main.py` 直接启动 RGB-D 采集界面，窗口标题为 **Airway3D**。保留实时预览、先闭口再张口采集、ArUco 检查、人脸检测、TUFA 关键点和语义分割。正常使用不需要提供 stage2 OBJ、prepared.npz 或历史采集数据。

当前算法为 **cross-v4-jaw-supported**：新版初始化与 stage1 → 增加观测支持的下颌约束的 stage2 → 一次中点细分 → 带细分适配及几何保护的 student 三维位移 → 额外修正不超过 0.1 mm 的位移场平滑。stage3 组件仍为 **cross-v3-bounded-smooth**。指标使用本次 stage2，结果页采用平滑法线显示 stage3。本目录包含源码与模型，运行无需两个来源项目或旧采集目录。

## 算法流程

1. **采集与预处理。** 分别保存闭口、张口 RGB-D 和内参，取得面颈点云、68 点二维／三维观测、分割标签。闭口必须取得有效 ArUco 喉结三维点，否则停止并保留数据。
2. **新版初始化。** 无尺度刚性对齐使用 17–47 号点，排除 17、18、19、24、25、26，至少 8 个有效配准点；8、21、22、33、50、51、52、56、57、58 的三维观测须有效。颈部初值为零，模型尺度保持米。
3. **第一阶段：纯 LM。** 固定 shape／expression，优化各帧旋转、平移、颈部／下颌屈伸。点到点、点到平面、外唇三维距离权重分别为 0.1、0.9、10。
4. **第二阶段：渐进 FLAME。** shape／expression 基数按 **20／5 → 50／10 → 100／30** 增长，各档继承参数；两帧共享 shape，各自保留 expression、pose、旋转和平移。使用点云项、原 10 个三维语义点（权重 2）、内唇二维距离（1e-5），shape／expression 正则各 1e-6。v4 另加入下述下颌轮廓项和分区点云权重；没有额外全体 51 点二维重投影或 pose 正则。
5. **保存 stage2、计算指标。** 保存原始 FLAME 网格、参数与拟合标点，随后计算指标。第三阶段使用独立副本，不回写 stage2。
6. **一次中点细分。** 每面拆为四面，保持原顶点顺序、坐标及三角形表面。标准 FLAME 从 5,023 顶点／9,976 面变为 20,018 顶点／39,904 面；不采用平滑细分。
7. **第三阶段：受保护的 student。** 两种状态分别使用各自深度、分割和标点，在细分基准上求三维位移 D，输出 `V = V_subdivided + D`。保留旧 student 动态最近三角形表面对应、深度置信度、Huber、语义约束和稀疏位移正则，并加入下述保护。

初始化、stage1 和 stage2 的参数调度来自 `subdivide_scan`；v4 明确修改了 stage2 的目标函数，不再声称与来源 stage2 输出相同。三维语义按两帧有效坐标平均，唇距按有效点对平均，共享 shape 只正则一次。对应面、重心坐标、法线在外层更新，LM 内层固定。shape／expression 使用固定姿态 LBS 解析雅可比，pose／外部旋转使用中心差分。

## 第二阶段：下颌轮廓与分区权重

新采集数据表明，偏圆的轮廓主要在 stage2 已存在；受限 stage3 无法修复较大的脸型偏差。v4 保持拟合维数、调度及停止条件，增加以下目标项：

1. **观测支持的轮廓约束。** 从二维点 2–14 中选择分割支持的下颌点：内侧应为皮肤，外侧应存在背景、颈部或衣物，避开头发／耳朵。至少 6 点才启用。只惩罚轮廓法向偏差，允许沿轮廓滑动，不把轮廓编号当作精确的三维解剖对应。
2. **鲁棒计权。** 权重 `jaw_contour_weight=0.5`；使用固定内部标点深度和焦距换算像素尺度，Huber 阈值 `0.003 m`。IRLS 权重外层更新、LM 内层固定。不直接使用下颌边缘可能受背景污染的三维深度点作为标点目标。
3. **点云分区。** 根据输入图像的皮肤标签和观测下颌线固定下颌内侧带。下颌带／其余面部／颈部的总权重预算为 **0.30／0.55／0.15**，组内按点数归一；空组重新归一。下颌带至少 20 点且轮廓点足够才启用，否则沿用均匀权重。点到点／点到平面的 0.1／0.9 比例不变。
4. **回退与记录。** stage2 初始偏航超过 20°，或分割／标点支持不足时，不强行增加轮廓约束。保存选点、分区数量、回退原因、初始偏航及轮廓法向残差。不指定“尖下巴”模板，不使用扫描监督。

两帧仍共享 shape，因此闭口、张口及颈部之间存在拟合取舍。当前验证显示三次闭口下颌贴合改善，但部分张口／颈部指标略差；单一正面 RGB-D 不能充分确定后侧下颌角，不能据此保证真实三维下颌完全准确。通过 `FitOptions(jaw_contour_weight=0.0, jaw_region_balance=False)` 显式传入 `run_pair(..., options=...)` 可关闭本次新增目标项。

## 第三阶段：细分适配与几何保护

细分增加自由度，没有增加深度信息。直接沿用旧版 8 mm 位移和宽松边长／方向限制，新增顶点可能追随噪声产生凹坑与折角。原先锚点补零列只保证初始位置一致，位移后原面插值点不一定仍位于子面表面。当前交叉版作以下调整：

1. **子面锚点。** 在原 FLAME 拓扑构造 51 个重心坐标锚点，再映射到各自所属子三角形；验证初始位置一致，通过显式 `landmark_operator` 传递。不把旧面片编号用于新面片，也不做全局 monkey patch。
2. **中点位移一致性。** 增加 `D_mid - (D_a + D_b)/2` 的平方惩罚，权重 16，抑制新增中点独立鼓包。约束的是位移，不平滑原始形状，也不永久锁死中点。
3. **短边保护。** 邻边位移差算子按 `0.002 / max(edge_length_m, 0.0002)` 缩放，再形成平方正则项，提高短边位移突变的代价。
4. **小步更新。** 总位移范数最多 **2 mm**，每次接受的更新范数最多 **0.25 mm**；总位移始终相对细分基准。旧版 8 mm 也是总位移限制，不是每轮步长。
5. **参考几何约束。** 相对细分基准限制边长比 0.8–1.25、面积比 0.7–1.4、面法线转角最多 20°、有符号相邻面折角变化最多 10°。先局部减步，再做能量回溯，不接受不安全候选。
6. **v3 受限位移平滑。** v2 student 求解及早停不变，最后对其位移 D 求一个小修正 X，最小化 `||X||² + ||L(D+X)||²`。仅在原可动区域求解，冻结全部语义约束涉及的顶点；不平滑 stage2 的绝对坐标。每顶点额外修正最多 0.1 mm，总位移仍最多 2 mm，所有几何条件继续检查。仅在目标下降时接受，否则保留 student 结果。最终输出并保存的是总位移 `D+X`，只相对细分基准应用一次。

只复用 `conservative_refine.ShapeGuard` 的几何检查，不执行完整保守法线位移算法。当前仍是 student 三维位移，可靠深度筛选、可见区域、语义保护和过渡区域沿用 student。

| 参数 | 当前值 | 含义 |
|---|---:|---|
| 平滑阶段 | 4 / 1 / 0.25 | 位移 Laplacian 正则，每档最多 8 轮 |
| edge_weight / anchor_weight | 0.05 / 0.01 | 邻边位移差／零位移正则 |
| 语义权重 | 40 | 原 student 二维射线与可靠深度约束 |
| 最大深度样本数／种子 | 16,000 / 42 | 固定抽取可靠深度点 |
| 对应门限／Huber | 20 mm / 2 mm | 保留旧 student 数据项 |
| 中点一致性权重 | 16 | 位移相对端点平均的平方惩罚 |
| 总位移／单次更新 | 2 mm / 0.25 mm | 当前交叉版新增限制 |
| student 步长早停 | 0.1 mm | 保留旧配置；保护导致的小步也可能触发 |
| student 平台判定 | 1%，连续 2 次，至少 3 次 | 当前有效对应 RMSE 的相对变化 |
| v3 位移场平滑强度 | 1 | `||X||² + strength*||L(D+X)||²` |
| v3 额外修正上限 | 0.1 mm | 相对 v2 输出；不是放宽 2 mm 总位移上限 |

这些条件抑制局部畸变，不构成全局自交检测或所有采集均无凹坑的保证。代价是保留部分 stage2 偏差、放弃部分噪声和真实细节。超过小幅修正范围的问题应排查采集和 stage2，不能仅靠放宽位移追着深度贴合。内部深度残差下降不等于真实精度提高。程序不含 VEGA、扫描监督或训练。

## 收敛配置

`core/fit_options.py` 保持新版配置：第一阶段最多 10 轮外层，第二阶段每档最多 20 轮，每轮最多 12 步 LM。梯度阈值 1e-10、参数步长阈值 1e-7、顶点变化阈值 1 微米；外层至少 6 轮后判断收敛。平台期要求相对变化小于 1e-4 且顶点变化小于 2 微米，连续 4 轮；完全拒绝更新不能视为收敛，每轮最多 6 次连续拒绝。这些是数值阈值，不是测量精度。

## 指标与显示

- stage3 前使用 stage2、拟合关键点、原观测及 ArUco 点计算指标；公式沿用原项目，未验证临床有效性。
- 实际调用链为 `PipelineWorker → run_pair → finish_meshes → cross_refine.refine_mesh`，闭口／张口分别处理，显示最终 stage3。
- 窗口标题为 Airway3D，中文字体使用微软雅黑。无相机或有效 ArUco 时显示错误，不默认使用旧数据替代采集。
- 默认采用顶点法线平滑着色，减少三角形明暗分块。`core.render.auto_render_mesh(..., smooth=False)` 可查看平面着色。着色不移动顶点，渲染前复制输入，显示效果改善不作为几何精度结论。

## 启动

使用 Python 3.10、兼容驱动的 CUDA PyTorch、Orbbec SDK／驱动。已验证环境见 `requirements-tested.txt`、`docs/environment.json`。TUFA 需要 CUDA；分割 ONNX Runtime 使用 CPU。

```powershell
cd D:\项目\sub_oldstage3
python main.py
```

也可双击 `start.cmd`。启动器依次查找 `AIRWAY3D_PYTHON`、本地 `.venv`、本机 `D:\softwares\miniconda\envs\airway3d\python.exe`，最后尝试 PATH Python。其他电脑可设置 `AIRWAY3D_PYTHON`；环境位置不属于源码或数据依赖。

新环境按 `requirements.txt` 安装，并执行 `python -m pip install chumpy --no-build-isolation`。安装匹配设备的 `pyorbbecsdk` wheel 和驱动，要求 USB 3、RGB 1280×720、硬件深度到彩色对齐。保留提供 ArUco 的 OpenCV contrib 功能。

## 输出

每次采集写到独立的 `output/<时间戳>/`：

```text
capture_pair.npz              两帧 RGB-D 与内参，失败也保留
prepared.npz                  点云、分割、标点和有效性
flame_parameters.npz          共享 shape；各帧 expression/pose/刚性参数
stage2_landmarks.npz          第二阶段拟合标点
fitting_report.json          拟合参数档位、迭代及停止原因
metrics.json                 第二阶段指标和单位
provenance.json               算法、输入／源码／模型资产哈希
report.json                   阶段运行状态、耗时和诊断
closed/mesh_closed_stage2.obj 原始闭口 FLAME
open/mesh_open_stage2.obj     原始张口 FLAME
closed/mesh_closed.obj        最终闭口交叉 stage3
open/mesh_open.obj            最终张口交叉 stage3
closed/vertex_offsets.npz     reference_vertices、faces、offsets
open/vertex_offsets.npz       各自的细分基准与相对位移
closed/stage3_report.json     位移、参数、几何检查和迭代记录
open/stage3_report.json
```

网格为**米**，深度图为**毫米**；相机转网格时 Y/Z 取反。位移只与细分基准相加一次，不能与未细分 stage2 数组直接相减。以 report.json 判断完成／失败状态。

## 验证

当前 v4 的三组双帧对照见 [下颌拟合对比页面](diagnostics/jaw_candidate/index.html)，方法、取舍和检查结果见 [v4 验证](docs/JAW_FIT_VALIDATION.md)。真实 UI 工作线程离线回归结果在 `diagnostics/jaw_integrated/checks.json`：实际执行 ArUco、stage1、stage2、细分和 stage3；分割／标点使用缓存，渲染使用测试替身，未验证硬件采集。原采集文件未覆盖。复现命令：

```powershell
python -X utf8 tools/check_jaw_jacobian.py --run output/20261025_210043_671653
python -X utf8 tools/validate_jaw_fit.py --runs output/20261025_204621_992883 output/20261025_205026_143947 output/20261025_210043_671653
python -X utf8 tools/check_jaw_pipeline.py --run output/20261025_210043_671653
python -X utf8 tools/build_jaw_review.py
```

以下 v1–v3 记录描述对应历史版本，不代表当前 stage2 与原输出相同。

历史 v1 验证保留在 `validation/checks.json`：当时与交叉参考 OBJ 的最大差异 8.66e-9 m，UI worker 离线完整流程通过。它属于旧版记录，不能代表当前 v2 与旧输出一致。

历史 v2 的双帧验证与完整回归分别在 `validation/cross_v2/`、`validation/cross_v2_full/`，详见 [v2 验证](docs/CROSS_V2_VALIDATION.md)。当前 v3 双帧对比、真实渲染与保存链路检查在 `validation/cross_v3_smooth/`，详见 [v3 验证](docs/CROSS_V3_VALIDATION.md)。验证必须显式提供路径，不是程序默认输入：

```powershell
python tools/check_v3_contracts.py
python tools/validate_cross_v3.py --run validation/cross_v2_full/runs/20261025_153220_481039 --render
python tools/check_stage3_pipeline.py --run validation/cross_v2_full/runs/20261025_153220_481039
python tools/build_v3_review.py --comparison validation/cross_v3_smooth/20261025_153220_481039
```

检查细分保持原表面、子面锚点、输入与 stage2 不变、位移只加一次、位移／步长限制和几何保护。v3 另检查语义锚点位置不变、额外修正不超过 0.1 mm、真实 pyrender 的平滑／平面着色均不修改输入。对比图使用相同视角与光照。固定深度样本的表面距离仅是贴合诊断，不是 VEGA 真值或解剖对应误差。此次没有重新验证硬件采集，也没有重跑未修改的 stage1/stage2；检查的是已有 stage2 到 stage3 的真实衔接。

当前 stage3 组件默认 v3；程序化调用 `core.cross_refine.refine_mesh(..., profile='v2')` 可复现无末端平滑的 v2 对照。旧源码快照在 `docs/archive/cross_v2_before_v3/`，本次 stage2 修改前快照在 `docs/archive/before_jaw_constraints/`。一次增加迭代与切向约束的候选未取得一致收益，未采用；其输出在 `validation/cross_v3/`，勿与已采用的 `cross_v3_smooth/` 混淆。

## 主要文件与来源

| 文件／目录 | 职责 |
|---|---|
| `main.py`、`start.cmd` | 采集入口 |
| `core/joint_lm.py`、`core/fit_options.py` | 新版 stage1/stage2 与配置 |
| `core/jaw_constraints.py` | 分割支持的下颌轮廓与固定点云分区 |
| `core/pipeline.py` | 指标隔离、阶段衔接和保存 |
| `core/cross_refine.py` | 细分、子面锚点、中点一致性及 stage3 参数 |
| `core/student/` | 隔离的 student 模块及新增保护接口 |
| `core/displacement_fairing.py` | 冻结语义支撑、0.1 mm 修正界限及位移场平滑 |
| `core/conservative_refine.py` | 复用 ShapeGuard；完整法线细化不执行 |
| `ui/` | 实时预览、采集、后台计算、显示 |
| `assets/`、`models/`、`third_party/` | 模型与第三方资源 |
| `docs/source_manifest.json` | 原始来源、压缩包差异及整合哈希 |
| `docs/asset_manifest.json`、`licenses/` | 资产哈希及许可 |

新版来源为 `D:\项目\subdivide_scan`；student 来源为 `D:\项目\Airway3D_0\new_version`，按 `jiaochastage3` 生成清单核实。`D:\Airway3D.zip` 参考结构与交互，其部分 stage2 与指定新版不同，未用于替换新版拟合。v2／v3 是明确修改的交叉版本，不再宣称复现原旧 student 输出。来源项目保持原样，模型使用条件以各自许可为准。
