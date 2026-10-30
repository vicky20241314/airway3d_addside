# 单相机 RGB-D 人头独立研究入口

本目录不改变 GUI、默认算法、VERSION 或共享标签。基线固定为
`3e409af77a4686b0cf435fff2230b90a8d26e11d`（v0.12.0）。
运行时资产、环境及结果放在仓库外；所有入口拒绝覆盖既有结果目录。
最终实测结果见 `RESULTS.md`（完成实验后生成）。

## 方法边界

A 使用作者 `linchaobao/hifi3dface` 的实际几何脚本。`run_a.py` 复制作者
RGBD/test1 样例、运行 step0–5，在 HD 纹理生成之前停止。仍包含作者几何
流程使用的 albedo 预拟合，不能把它称为纯深度算法。公开 Python 版没有
论文优化 C++ 版的速度保证。头部补全不是扫描精度证据。

B 为自主的离线试验实现，不是完整 FaceCap 复现。原文的 augmented
blendshapes 和 UV Bump/Deviation 图启发了“表情与身份分离、消除运动后融合”
的设计。这里改用 FLAME、稀疏顶点法向偏移和 PyTorch 优化，不复用其
Kinect/OpenCL 实现，不声称实时性、UV 分辨率或论文相同精度。

`prepare.py` 读取真实 raw_sequence.npz 和同段独立逐帧关键点检测，重新用
原始深度反投影。相邻上半脸 KLT + RGB-D 刚性估计用于初始化；失跟允许
关键点重定位。颈部使用独立掩码跟踪，不参与身份拟合，也不要求头颈变换一致。
没有颈部可靠特征时记录降级，当前不拟合独立颈部身份网格。

`fit_b.py` 共享 50 维身份参数；每帧各有 10 维表情、三轴下颌和六自由度
相对姿态。质量、角度新颖性和时间间隔选择关键帧，仅关键帧更新身份；
其余有效帧仍估计运动。时间约束仅跨同一短序列；跨数秒的采集缺口不假装
有连续跟踪。先拟合姿态/下颌，再解锁身份/表情，最后按可见性、语义及
深度一致性筛选公共法向偏移，采用中位数融合。绝不直接平均不同表情深度。
偏移限幅 3mm 是独立试验约束，未放宽原项目几何保护。

## 环境

Windows B：Python 3.10、Torch 2.0.1/CUDA 11.7、NumPy 1.26.4、SciPy 1.11.4，
依赖清单 `requirements-b.txt`。数据准备另外复用已有项目的 uniface 语义分割
模块。优先克隆已验证环境到仓库外，再验证实际 import 版本；Conda 元数据
可能与 pip 覆盖后的实际 NumPy 不符，必要时仅对新环境 force-reinstall。

A：WSL Linux、Python 3.6.15、CPU TensorFlow 1.15.5，见
`requirements-a-cpu.txt`。这是针对当前设备的兼容环境，与作者完整旧锁文件
有明确差异；不承诺作者的 GPU 性能。`build_a.py` 从实际 TensorFlow
sysconfig 获取编译参数，替代 install.sh 中作者个人机器的绝对路径。

上游代码固定 d5accd138359ad534662805cd710f58de4ee4c74。源码 MIT；模型仅研究
用途，商业用途需联系作者；CelebAMask-HQ 相关组件另限非商业研究教育。
FLAME 使用沿用项目已提供资产及其许可，不随本研究新增分发模型。
FaceCap 两仓库未见顶层明确许可文件，因此这里只核查，不复制代码。

## 可复现命令（PowerShell）

以下路径为本次执行位置；复现时为输出选择新的目录名。

```powershell
$wt = 'D:\proiect\rgbd-head-research-20261030'
$run = 'D:\proiect\rgbd-head-research-runtime-20261030'
$py = "$run\env-b\python.exe"
$capture = 'D:\proiect\newstage2_oldstage3\output\capture_mv_20261030_095649_134226'
& $py "$wt\research\head_rgbd\audit.py" --source "$run\hifi3dface" --capture $capture --output "$run\audit-new" --probe-assets
& $py "$wt\research\head_rgbd\prepare.py" --capture $capture --output "$run\sequence-new" --model-cache "$run\models\uniface"
& $py "$wt\research\head_rgbd\run_baseline.py" --session "$capture\complete\session.json" --output "$run\baseline-new" --model-cache "$run\models\uniface" --device cuda
& $py "$wt\research\head_rgbd\fit_b.py" --sequence "$run\sequence-new\sequence.json" --output "$run\b-new" --device cuda --steps 240 --points 1200 --keyframes-per-segment 3
& $py "$wt\research\head_rgbd\fit_b.py" --sequence "$run\sequence-new\sequence.json" --output "$run\b-front-new" --device cuda --steps 240 --points 1200 --keyframes-per-segment 3 --front-only
& $py "$wt\research\head_rgbd\evaluate.py" --sequence "$run\sequence-new\sequence.json" --capture-session "$capture\complete\session.json" --baseline "$run\baseline-new" --b "$run\b-new" --b-front "$run\b-front-new" --output "$run\comparison-new"
& $py -m unittest discover -s "$wt\research\head_rgbd" -p test_geometry.py -v
```

上述重建命令依次执行，不能同时测 GPU 耗时。`prepare.py --copy-cache-from`
可把已有 uniface 缓存复制到任务自己的缓存；禁止让模型下载器写入原项目。

A 下载及样例：

```powershell
& $py "$wt\research\head_rgbd\fetch_assets.py" --output "$run\downloads-new" --assets sample model resources --max-seconds 1800
# 若大文件超时，只在 Content-Range 验证通过后续传：
& $py "$wt\research\head_rgbd\resume_asset.py" --partial "$run\downloads-new\model.partial" --asset model --report "$run\resume-new.json"
wsl -d Ubuntu-24.04 -- bash -lc '/mnt/d/proiect/rgbd-head-research-runtime-20261030/env-a/bin/python /mnt/d/proiect/rgbd-head-research-20261030/research/head_rgbd/build_a.py --source /mnt/d/proiect/rgbd-head-research-runtime-20261030/hifi3dface --output /mnt/d/proiect/rgbd-head-research-runtime-20261030/build-new'
wsl -d Ubuntu-24.04 -- bash -lc '/mnt/d/proiect/rgbd-head-research-runtime-20261030/env-a/bin/python /mnt/d/proiect/rgbd-head-research-20261030/research/head_rgbd/run_a.py --source /mnt/d/proiect/rgbd-head-research-runtime-20261030/hifi3dface --output /mnt/d/proiect/rgbd-head-research-runtime-20261030/a-sample-new'
```

## 数据与坐标契约

输入 RGB 和深度必须已物理对齐；仅尺寸一致不能证明对齐准确。当前采集格式
深度为 uint16 毫米，每帧存 [fx,fy,cx,cy]，米制点为 X右/Y上/Z后。
`adapt_a.py` 转换作者特有 `(R*255+G)/10 mm` PNG，并作往返检查；输出
calibration.json 写明转置前后内参和坐标转换。作者源码硬编码 iPhone 内参、
前处理转置及 86 点检测尚须实际接通；格式转换成功不能叫 A 真实数据复现。
禁止将我们的 68 点直接冒充作者 86 点。

闭口和张口网格的 report.references 绑定真实原文件、帧号、时间戳、姿态、
表情和 FLAME 68 点标号，网格 OBJ 及标点均为对应参考相机坐标的米制几何。
未新增临床指标定义，不能拿本研究网格直接替换现有临床指标输出。

## 验收规则

evaluate.py 比较相同四个真实原始参考帧，报告相机轴向深度残差、射线覆盖率
及 5mm 内支持率。这些是训练数据诊断，不是独立表面误差。
没有独立外参、独立表面或人工轮廓时，正/侧面表面误差、轮廓误差、独立有效
覆盖率、正面退化验收全部明确为未验证。正面对照用于发现明显回归，不构成
严格的独立消融结论。完整头部中的未观测区域是模型先验补全。

原论文：[腾讯项目](https://tencent-ailab.github.io/hifi3dface_projpage/)、
[CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Thomas_Augmented_Blendshapes_for_CVPR_2016_paper.pdf)、
[2020 FaceCap](https://arxiv.org/html/2004.10557)。
