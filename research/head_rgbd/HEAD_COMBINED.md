# A3 人脸与原项目整体头部的实际组合

已实际导出闭口和张口的整体头部米制网格，包含耳部、头顶、后脑和颈部。
**未观测部分来自原项目的 FLAME 先验，不能称为整个人头的真实扫描。**
本次为显式独立适配，读取固定版本已有结果，未修改原项目默认入口或其他
agent 的未提交修改；未试用它们尚未提交的算法改动。

## 方法与许可

输入是本机已有 A3 观测人脸及原项目既有整体头部结果，分别试验：

- `b`：已有 B 的共享身份、逐帧姿态/表情/下颌**粗头部**，绑定实际参考帧。
- `baseline`：原项目固定 v0.12.0 / `3e409af...` 的 stage2 头部，核对采集 session
  SHA 和真实正面代表帧；不是重新运行默认 GUI 或最新未提交改进。

只读复用 Python 3.10.22、NumPy 1.26.4、SciPy 1.11.4；Pillow 仅本地 QA。
没有加载 FLAME pickle、TUFA、分割或感知权重，没有 CUDA 上下文或相机采集。
**读入 FLAME 派生网格仍受模型与数据条件约束**，不是借格式转换解除限制。
用户已明确确认官方学术许可及本地非商业用途，具体证据、来源、限制和必要
署名见 `HEAD_COMBINATION_LICENSES.md`。商业和再分发未获本次确认。

## 几何处理

1. 旧项目的 X右/Y上/Z后坐标用 `diag(1,-1,-1)` 转成 A3 的 X右/Y下/Z前，
   单位保持米；用 A3 最终刚性 T 进入共同参考相机。没有新的尺度优化。
2. 一次线性中点细分：5023 顶点/9976 面变为 20018 顶点/39904 面。
   保留原三角面表面位置，**细分不是增加实测细节**。
3. A3 数据支持、方向、背面和头部自身 Z-buffer 筛选约束；对可靠脸面求解
   点面对应与图拉普拉斯位移正则，未观测后脑等区域固定。
4. 从粗头部开始，总新增修正 <=3mm，每步 <=0.75mm；不叠加旧 B 可选的
   3mm 法向细化。相对面积比、法向、边长和翻转门限未放宽。细长三角形
   及一环邻域额外锁定，闭口/张口共同保护。
5. 闭口求得的上半脸身份修正场也固定应用于张口；下半脸有各自参考状态修正。
   共享的是新增身份修正场，不要求原 B 的逐帧表情顶点完全相同。
6. 导回各自真实参考相机，输出无纹理 OBJ、同阶段命名标点及支持/来源侧车。
   原模型眼球等分部保留；没有切下人脸后拼成双层重叠面。

观测约束、正则过渡和先验区按顶点记录；OBJ 的组用于粗略显示，混合面应以
`provenance.npz` 的逐顶点标记为准。支持度是同一批拟合观测的深度一致性，
不是独立扫描精度或完整头部覆盖率。

## 本地最终产物

根目录 `D:\proiect\rgbd-head-a-adapt-runtime-20261030`：

- `head-b-final-v2/closed/head_combined_m.obj`
- `head-b-final-v2/open/head_combined_m.obj`
- `head-baseline-final-v2/closed/head_combined_m.obj`
- `head-baseline-final-v2/open/head_combined_m.obj`

各状态同时包含 `head_prior_m.obj`、`provenance.npz`、`provenance.json`、
`landmarks_combined_head.json`；整次有完整 `report.json`、执行源码快照和
`whole_head_geometry.png`（无纹理正/侧/后查看）。真实引用帧为
closed_front_1 / open_front_2，源文件/帧号/时间戳及最终相机 T 在本地侧车。
命名标点不等同于 FLAME 68/作者86点或已验证临床指标。

每个最终头部为 20018 顶点、39904 三角形、3 个原有组件；保留两个边界环、
124 条边界边。零非流形边，零翻转。**整体头部形状已具备，但不是 watertight**，
没有把原颈口/口腔边界封成已测量表面。

| 先验 | CPU 时间 | 闭口/张口最大新修正 | 共享修正误差 / 固定区位移 |
|---|---:|---|---|
| B 粗头部 | 88.93s | 0.120 / 0.517mm | 0 / 0 |
| 原项目基线 | 104.99s | 0.134 / 0.594mm | 0 / 0 |

时间包含新适配的数据读取、拟合、导出和同帧 CPU 诊断，不含历史头部拟合、
A3 准备或最终预览；不是与原 GPU 管线的等精度性能比较。全部顺序在 CPU 上运行。

B 组合闭口/张口分别 5520/5534 顶点收到直接观测约束且有多帧支持；
12335/12047 顶点保持固定先验，其他为正则过渡。未观测后脑、头皮等不作精度保证。
修正明显受严格几何保护限制，不能将整体头形完整误读为 A3 已恢复高保真整头。

## 训练数据诊断与限制

四个真实代表帧、相同 ROI、原像素步长4，比较**同一头部与同一姿态**适配前后：

| 真实帧 | B 粗先验→组合，中位 / P95 mm | 基线先验→组合，中位 / P95 mm |
|---|---|---|
| 闭口正面 | 1.485→1.352 / 6.757→6.475 | 1.686→1.525 / 8.459→8.118 |
| 闭口左侧 | 1.782→1.660 / 5.604→5.438 | 1.880→1.749 / 5.833→5.647 |
| 闭口右侧 | 1.525→1.448 / 6.529→6.247 | 2.220→2.126 / 7.624→7.468 |
| 张口正面 | 1.635→1.408 / 66.602→61.092 | 2.159→1.816 / 56.840→56.599 |

这是相机轴向训练残差，不是独立欧氏表面误差。B 输入是粗头部，未叠加历史
法向细化；斜侧姿态采用 A3 的同一位姿，并投影同一个参考状态头部，因此本表
不能直接与旧 B“每帧表情/姿态＋细化”的诊断数值混为一谈。
张口口腔仍有大尾部误差，不能声称口内或下颌准确。完整先验投影也不等于完整
实测覆盖。独立扫描、轮廓、外参和后脑真实深度均缺失，整头准确性未验证。

短流程 `head-b-trial-v1` 用4步，仅用于接口检查；正式为8步。初次全局保护过早
限制局部变形，后增加细长三角形锁定，但仍保留明显的拟合能力限制，没有通过
放宽保护追求较低损失。六项组合契约测试通过；A3 原13项测试为先前已验证结果。

## 复现命令（PowerShell，输出用新目录）

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$py = 'D:\proiect\rgbd-head-research-runtime-20261030\env-b\python.exe'
$run = 'D:\proiect\rgbd-head-a-adapt-runtime-20261030'
Set-Location D:\proiect\rgbd-head-a-adapt-20261030
& $py -B -m unittest discover -s research/head_rgbd -p test_head_adapter.py -v
& $py -B research/head_rgbd/run_head_combined.py --a3 "$run\formal-v4" --manifest "$run\tracked_manifest.json" --legacy-runtime D:\proiect\rgbd-head-research-runtime-20261030 --authorization "$run\head-research-authorization.json" --output "$run\head-b-reproduce" --head-source b --subdivide 1 --iterations 8 --preview
& $py -B research/head_rgbd/run_head_combined.py --a3 "$run\formal-v4" --manifest "$run\tracked_manifest.json" --legacy-runtime D:\proiect\rgbd-head-research-runtime-20261030 --authorization "$run\head-research-authorization.json" --output "$run\head-baseline-reproduce" --head-source baseline --subdivide 1 --iterations 8 --preview
```

`head-research-authorization.json` 是本机已有真实授权声明记录，不发布；不能通过
任意填写布尔值取得第三方权利。入口要求模型哈希匹配、学术许可确认、明确本地
非商业用途以及不允许分发。输出目录拒绝覆盖。无授权/不同资产时不能直接套用。

机器摘要 `head-combination-summary.json` 只含聚合统计。实际头部、模型、标点、
姿态和图像均未上传 GitHub 或其他重建服务。默认算法、版本号和标签未修改。
