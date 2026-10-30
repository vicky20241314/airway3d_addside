# 单相机 RGB-D 米制观测人脸（A3 独立入口）

这是基于方案 A 多视角几何思想的独立研究原型。**不是作者方法复现，不是高保真
扫描精度验收通过，也不是全自动产品。** 不需要作者模型、FLAME、TUFA、分割、
识别或感知权重。使用已有真实 RGB-D 和本地人工种子，输出没有纹理的人脸网格。

结果见 `A3_RESULTS.md`、`experiment-a3-summary.json` 和 `A3_COMPARISON.json`。
许可、原始声明与未明确项见 `LICENSES_A.md`、`A_DEPENDENCIES.json`、
`THIRD_PARTY_A3.md`、`A_LICENSE_QUESTIONS.md`。所有真实人脸产物留在仓库外。

## 与旧方法的关系

| 层 | 来源/处理 |
|---|---|
| 原版 A | 作者固定 SHA d5accd138359ad534662805cd710f58de4ee4c74；仅保留历史证据 |
| 旧兼容补丁 | CPU 图布局、旧 pickle、UV 越界、内参/方向；原文件未修改，A3 不调用 |
| A3 新几何 | 人工 2D/反投影 3D 标点锚定，SE(3) 上半脸点面优化，观测一致性融合，有界几何平滑 |
| A3 运动/表情处理 | 共享上半脸身份；每帧独立刚性位姿；下半脸仅融合参考帧所在真实短段内且深度一致的观测 |
| A3 逐帧入口 | 手工点在实际 RGB 中通过归一化块相关跟踪；无学习模型，不能跨长采集缺口插帧 |
| 旧基线/B | 只读取已存在网格/参数做本地数值比较，不运行模型；原许可及数据条件不因此改变 |

没有替换默认 GUI/算法，也没有改 VERSION、标签或 main。已有项目若需接入，
应显式调用此入口并读取侧车，不得把新 OBJ 直接冒充原临床标点拓扑或默认输出。

## 环境与路径

- 独立 worktree：`D:\proiect\rgbd-head-a-adapt-20261030`
- 分支：`codex/rgbd-head-a-adapt-20261030`，从研究提交 `159a894...` 分出。
- 独立运行目录：`D:\proiect\rgbd-head-a-adapt-runtime-20261030`
- 只读复用解释器：`D:\proiect\rgbd-head-research-runtime-20261030\env-b\python.exe`
- 实测 Python 3.10.22 / NumPy 1.26.4 / SciPy 1.11.4，见 `requirements-a3.txt`。
- 可选本地 QA 用 Pillow 11.1.0；重建不需要 Pillow。

没有重新安装环境或模型，没有修改作者副本，没有启动相机流。重建为 CPU，
不分配 CUDA 显存，也不运行其他 agent 的任务。使用 `-B` 防止写入共享字节码缓存。
无需下载已有资产；许可审计重新验证了三个归档的 SHA。

## 复现当前真实数据

命令在独立 worktree 内执行。输出目录必须不存在；下列 `reproduce-*` 路径是新的，
不能拿已有 `formal-v4` 等结果目录重跑覆盖。人工清单保留在本机，未上传。

```powershell
Set-Location D:\proiect\rgbd-head-a-adapt-20261030
$env:PYTHONDONTWRITEBYTECODE = '1'
$py = 'D:\proiect\rgbd-head-research-runtime-20261030\env-b\python.exe'
$run = 'D:\proiect\rgbd-head-a-adapt-runtime-20261030'
& $py -B -m unittest discover -s research/head_rgbd -p test_a3.py -v
& $py -B research/head_rgbd/expand_a3_frames.py --manifest "$run\manual_manifest.json" --output "$run\reproduce-tracked.json"
& $py -B research/head_rgbd/run_a3.py --manifest "$run\reproduce-tracked.json" --output "$run\reproduce-full" --stride 2 --icp-iterations 30 --skip-diagnostics
& $py -B research/head_rgbd/run_a3.py --manifest "$run\reproduce-tracked.json" --output "$run\reproduce-front" --stride 2 --icp-iterations 30 --skip-diagnostics --front-only
& $py -B research/head_rgbd/check_a3.py --result "$run\reproduce-full" --front-control "$run\reproduce-front" --manifest "$run\manual_manifest.json" --legacy-runtime D:\proiect\rgbd-head-research-runtime-20261030 --output "$run\reproduce-checks" --preview
```

这些命令应顺序执行，每条检查 `$LASTEXITCODE`；任一失败时先看日志，不要继续
把不完整结果当成功。短流程用 `--icp-iterations 5 --skip-diagnostics`，正式为每个
固定门限最多 30 次（3 个门限，允许提前收敛）；与作者 10/100 次优化不是同一预算。

重新核对许可与归档（只读取共享资产，写到新的任务目录）：

```powershell
& $py -B research/head_rgbd/audit_a3.py --shared D:\proiect\rgbd-head-research-runtime-20261030 --output "$run\reproduce-license-audit" --notices "$run\reproduce-license-notices"
```

## 输入契约与新数据

`manual_manifest.json` schema 为 `head_rgbd.a3_manual.v1`，必须明确
`depth_unit: mm`、RGB/depth 对齐证据及本地处理授权。示例骨架
`a3_manifest.example.json` 不含任何真实人脸坐标，**不能直接用于效果演示**。

每个真实 raw NPZ 包含 `rgb[N,H,W,3] uint8`、`depth[N,H,W] uint16`（毫米）、
`intrinsics[N,4]`（或固定 `[4]`）及 `timestamps_s[N]`。
每个种子声明原文件、实际 source_frame、front/left/right/可选 up、closed/open、
人工 ROI 多边形、命名 2D 点和上半脸分界行。ROI 是裁剪输入，不是独立人工轮廓真值。
参考帧必须是实际输入。至少四个非退化且深度有效的稳定上半脸对应点用于位姿初始化；
这里采用眼角、鼻梁和鼻尖，口唇及下颌点另作侧车和诊断。

`expand_a3_frames.py` 每段只接受一个直接标注种子，用 21×21 灰度块在 ±8px 内
匹配实际帧，NCC >=0.85，边界峰拒绝；不足四个稳定点时整帧记录为排除。
只允许段内相邻间隔 <=0.3s、总长 <=0.5s，不能据此声称支持长视频或跨缺口时序。
传播点的来源、分数、失败均保存在本地清单。该清单也包含私密人脸标点，禁止提交。

重建支持 1..32 个唯一物理帧，未固定四视角；多状态参考仍需真实帧。
缺 up 明确记录降级，open 侧面缺失也单列，绝不复制正面。
配准失败可排除该帧继续其余流程；必要参考帧失败则明确终止。

## 几何与坐标链

1. 直接读取每帧实际内参和原始深度；不读取 sequence-v2 中神经网络点/掩码。
   RGB/depth 栅格尺寸一致仅为必要条件，物理对齐仍未独立标定。
2. `depth_m = uint16_mm / 1000` 是输入编码契约；不是按 OBJ 包围盒猜单位。
   采样 `[::stride,::stride]` 同步将 fx/fy/cx/cy 除以 stride，无图像转置、无深度插值。
3. 采用传感器相机坐标 **X右/Y下/Z前，米**：
   `p=[(u-cx)z/fx, (v-cy)z/fy, z]`。
4. `T_reference_from_camera` 为列向量 SE(3)：`p_common=R*p_frame+t`。
   旋转正交，平移为米，所有尺度固定 1；不优化逐帧或共享相似尺度。
5. 上半脸使用法向一致、距离门限、鲁棒点面项及 3D 标点锚，逐帧估计头部运动。
   损失按对应数归一化，固定步长/角度保护；没有放宽旧项目保护去掩盖失败。
6. 投到真实闭口参考相机的有序深度层，执行前表面 Z-buffer、无效深度、ROI、
   背向和深度一致性筛选。上半脸共享；下半脸仅用该状态参考所在同一真实短段，
   相对参考 <=1.5mm。参考孔洞须有 >=2 实测一致支持，不能以零深度作观测。
7. 多帧支持的局部几何细化最大偏移 0.8mm，不跨共享身份/表情分界，不凭空封孔。
   三角形边长须 <6mm，导出最大连通观测脸表面，剔除孤立噪声/牙齿片并记录数量。
8. 每个状态通过最终刚性逆变换导回**自己的真实参考相机**。闭口/张口都导出
   coarse/refined 网格与同阶段标点，不套用作者原生 OBJ 规范空间的尺度。

身份上半脸共享；下半脸采用参考状态，是保守的残差兼容处理，**没有完整表达式、
下颌关节或独立颈部模型**。颈部被 ROI 排除，其运动不会导致整脸重建中断。
未将不同头颈变换硬约束为同一个。它不能替代 B 的参数化表情/下颌功能。

## 本地产物和显式适配接口

每次输出包含 `report.json`（输入/清单/代码 SHA、帧选择、最终 T、预算、降级、
环境、时间与内存）以及每个 `closed/`、`open/`：

- `face_coarse_m.obj`、`face_refined_m.obj`：米制、无纹理、开放的观测表面。
- `landmarks_coarse.json`、`landmarks_refined.json`：命名人工点、原深度有效性、
  对应同阶段网格点与参考帧、原文件/帧号/时间戳。不是 68/86 拓扑，也非临床指标。
- `support_coarse.npz`、`support_refined.npz`：顶点/三角形、实际融合贡献数/位图、
  再投影观测支持数/位图、逐帧有符号轴向残差、共享身份区域和真实 frame_ids。
  **观测支持数与实际融合贡献数不同，不能混用。**
- `depth_and_validity.npz`：米制粗/细深度、显式有效掩码、所有真实深度层、内参、
  融合贡献以及最终输出变换。

适配代码应先核对 `schema`、`unit`、轴向、参考帧和阶段，再读取网格及同阶段侧车。
若接入旧项目 X右/Y上/Z后坐标，显式用 `diag(1,-1,-1)`；这是旋转，尺度不变。
不得直接替换临床关键点定义，也不能以网格整体 Z 跨度代替真实面部厚度测量。

`check_a3.py` 对当前闭/张口数据验证 OBJ 复读、索引、有限值、支持度、最大边长、
连通性和共享上半脸坐标；用共同人工 ROI 比较历史基线/B/新 A3，另报告所有方法
均渲染到的共同射线残差，避免遗漏区域导致“误差更小”的错觉。
对新数据可单独导出，不具备历史基线/B 时不能直接使用这个比较命令。

所有残差都是拟合数据的轴向诊断；未观测头部没有补全。缺少独立扫描、人工轮廓
真值与独立外参，不能报告独立表面误差或“正面没有退化”。研究运行与商用/再分发
权限分别报告，实际输入/模型/网格不随 Git 分支发布。
