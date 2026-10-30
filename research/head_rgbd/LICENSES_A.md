# A 路线许可审计（2026-10-30）

核查对象为固定作者版本 `d5accd138359ad534662805cd710f58de4ee4c74`，
已有研究基点 `159a894dcfb0760633723092133a2964f1fd6490`。结论按资产分开，
不提供无侵权、商业可用或数据权利保证。逐项清单见 `A_DEPENDENCIES.json`；
原始通知保存在 `third_party_notices_a3/`，本地完整取证记录在独立运行目录的
`license-audit/audit.json`。归档逐字节哈希重新验证通过，没有重新下载。

## 路线决定

**采用 A3：独立的、无模型的多视角 RGB-D 观测表面原型。**
作者方法的启发是选择可靠真实视角、用深度和标点约束几何、区分几何与纹理。
本实现不复现作者 3DMM/可微渲染/反射率模型，不继承其精度和速度结论。
数学使用 SE(3) 点面配准、人工标点锚定、深度一致性筛选、中位融合和有界平滑。
未复制作者、FaceCap、FLAME 或 TUFA 的算法源码。

上一轮拟选的 A1+A2 不作为本次完成路线：研究用途的笼统说明不能证明
resources 内每个权重、头部模板、训练数据及商业输出均已获授权。A1/A2 的
相关模型保持隔离。旧代码、兼容补丁和既往样例结果仅作历史证据。

## 作者代码、资产与旧项目的区别

| 项目 | 核查证据 | 研究/商业/分发结论 | 本次处理 |
|---|---|---|---|
| 作者主代码 | 固定 SHA 的 `LICENSE.md` MIT；`readme.md` 另有 research-only disclaimer | MIT 明文有软件复制修改和销售许可，但研究声明的适用边界仍有歧义；不能外推模型 | 不复制或运行；保留完整许可证和署名 |
| tf_mesh_renderer | 作者 third_party/readme 指定 `a6403fbb36a71443ecb822e435e5724550d2b52b`，Apache-2.0 | 保留许可、适用 NOTICE、署名、标记修改；代码允许商业用途 | 不加载；无需新编译 |
| pix2pix、N-ICP | 作者 `LICENSE.md` 中 MIT 归属 Christopher Hesse / Charlie Nash | 代码许可不覆盖任意外部训练数据与权重 | 不加载 |
| Tencent AI-NEXT 形状/反射率基 | `3DMM/readme.md`；model.zip 内 readme | 仅研究，要求论文引用；商业须另行联系；模型及输出的再分发边界未完整说明 | 不加载、不转换、不分发 |
| resources.zip：86/68 点、检测、分割权重 | 归档 readme 明确研究用途、不得分发；各权重独立训练来源和条款不足 | 存在研究范围说明，不能认定所有上游授权已覆盖；商业未明确 | 不加载；已有网络标点/分割结果不作为 A3 输入 |
| BFM2009、头部模板、UV/拓扑资源 | resources 归档，未找到逐资产授权链 | 需要逐项来源、版本、模型输出/分发授权 | 全部隔离，不做头部补全 |
| VGG-Face 代码/权重 | 作者 third_party/readme；ZZUTK repo 指向 MatConvNet 预训练 vgg-face.mat | 中间仓库未提供完整可用权重授权链，不能从作者 MIT 推断 | 不使用识别、感知或身份损失 |
| CelebAMask-HQ | 作者 LICENSE 明示非商业研究教育限制 | 不是宽松商业资产；还需区分代码、训练集、权重 | 不运行 |
| 作者样例 | sample.zip/readme.txt：部分互联网照片，研究算法测试、禁止分发 | 已有十步检查是历史研究证据；不是完整百步质量复现；不分发照片或输出 | 只复用先前记录，未重跑 |
| 旧 FLAME 包装代码 | 本地 third_party/flame/LICENSE 为 MIT | 软件许可不能覆盖 `flame2023.pkl` | 未导入旧模块 |
| 旧 FLAME 模型 | 官方 modellicense 区分学术模型与 FLAME 2023 Open | 学术模型非商业且限制分发；现有 pickle 未证明为 Open 版 | 不载入模型；仅本地读取既有输出比较，输出权利未清理 |
| TUFA 代码、提示/权重/检测器 | 本地 GPL-2.0；README 指向 WFLW、DETR、libfacedetection 等 | 代码商业使用不等于无条件封闭产品分发；权重和数据另审 | 完全不导入，不复用其关键点 |
| 旧分割入口/uniface 及模型 | 旧 prepare.py 经项目 mask_detection 调用 | 外层代码标签不能证明权重与数据商业条款 | 不使用 face_mask、neck_mask 或神经网络检测点 |

模型格式转换、蒸馏、导出网格不会自动解除原许可；本次没有进行这些操作。
没有替换成另一套未核实模型。没有联系任何作者，也没有代用户作承诺。

## 实际运行的依赖

| 依赖 | 实测版本/来源 | 条款与义务 | 研究、商业及分发边界 |
|---|---|---|---|
| CPython | 3.10.22，已有 env-b | PSF-2.0 及随附历史许可；保留通知，修改解释器时记录修改 | 软件条款允许商业；未打包解释器 |
| Anaconda 打包的现有解释器环境 | env-b，发行元信息见审计 | Python 软件许可与发行服务/组织合同不同 | 未检查用户与供应商的账户/合同资格；不分发环境，不新增安装 |
| NumPy | 1.26.4，实装 dist-info | BSD-3-Clause；源码/二进制通知与不背书要求 | 代码许可允许商业；保留完整二进制随附许可 |
| SciPy | 1.11.4，实装 dist-info | BSD-3-Clause；同上 | 同上 |
| OpenBLAS / LAPACK | NumPy DLL 0.3.23-293-gc2f4bdbb / SciPy DLL 0.3.20-571-g3dec11c6 | 随 DLL 的 BSD 及归属条款 | 允许商业，二进制分发需完整通知；DLL 哈希已记录 |
| GCC 运行库 | 上述 DLL 的 gcc_10_3_0 构建 | GPL-3.0 + GCC Runtime Library Exception | 例外适用条件须按具体二进制核查；本次只本地运行，不重新打包 |
| libquadmath | 安装包许可列示，实际链接范围未逐符号确认 | LGPL-2.1-or-later | 可依条件商用；分发须保留许可、对应源码/重链接等适用义务；本次无二进制分发 |
| NumPy 内嵌组件 | lapack-lite / dragon4 / libdivide；tempita 是构建来源组件 | BSD / MIT / Zlib，原始分发通知中逐项列出 | 保留原通知；本次未构建或修改这些组件 |
| Pillow | 11.1.0，仅本机 RGB 检查及无纹理 QA PNG | MIT-CMU，保留著作权与许可，不未经许可用作者名宣传 | 核心许可允许各种用途；PNG/JPEG 编解码二进制打包未作发布审计，不打包 Pillow |
| Windows 系统 DLL | 已有系统的 kernel32/psapi，只作峰值进程内存统计 | 系统软件条款，未分发 | 未核查设备授权合同，不复制系统组件 |

新重建运行时实际只依赖 Python、NumPy、SciPy；Pillow 是可选 QA 工具。
无 PyTorch/TensorFlow/CUDA 上下文，无网络下载，无模型依赖。用 `python -B`
及 `PYTHONDONTWRITEBYTECODE=1` 复现，防止导入产生共享环境字节码缓存。
未安装、升级或删除共享环境依赖。

## 数据与输出

用户明确指示本机处理已有自采序列，可追溯到原始 uint16 毫米深度及逐帧内参。
此指示不等同于可核验的受试者生物识别数据商业/公开传播同意书。
人工 ROI 和命名标点是本次直接观察真实 RGB 后新增的本地输入，不来自神经模型。
源数据、标点、网格和 QA 图均留本地，不随 Git 提交。

仅提交新代码、文档、许可通知和不含生物识别几何的聚合诊断摘要。
旧 FLAME/B/基线结果只在本地读取进行研究比较，既有输出仍受原资产和数据
条件约束，未声称其可商用或再分发。A3 新输出不含模型补全，但仍有数据主体、
采集合同、专利等未完成的权利审查；独立实现本身不证明这些权利不存在。

## 原始证据

- [作者固定 LICENSE](https://github.com/linchaobao/hifi3dface/blob/d5accd138359ad534662805cd710f58de4ee4c74/LICENSE.md)
- [作者固定 3DMM 条款](https://github.com/linchaobao/hifi3dface/blob/d5accd138359ad534662805cd710f58de4ee4c74/3DMM/readme.md)
- [方法与论文入口](https://tencent-ailab.github.io/hifi3dface_projpage/)
- [VGG 中间仓库](https://github.com/ZZUTK/Tensorflow-VGG-face)
- [FLAME 学术/Open 的区别](https://flame.is.tue.mpg.de/modellicense.html)
- [NumPy 固定许可](https://github.com/numpy/numpy/blob/v1.26.4/LICENSE.txt)
- [SciPy 固定许可](https://github.com/scipy/scipy/blob/v1.11.4/LICENSE.txt)

研究可执行范围已作组件隔离；**未来商用和产品分发审查未完成**。
需要的具体授权材料见 `A_LICENSE_QUESTIONS.md`。这里没有“保证无侵权”的结论。
