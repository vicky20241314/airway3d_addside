# 来源、归属与修改说明

研究思想来源：Bao 等，*High-Fidelity 3D Digital Human Head Creation from RGB-D
Selfies*, ACM TOG 2021。项目链接和固定版本见 `LICENSES_A.md`。
本实现是 A3 独立几何原型，不是作者实现、论文完整复现或作者认可的产品。

新增文件 `a3_geometry.py`、`run_a3.py`、`expand_a3_frames.py`、`check_a3.py`、`audit_a3.py`、`test_a3.py`
以及 `record_a3_summary.py` 为本任务新实现。没有移除第三方署名后将作者方法代码冒充原创。
已有 `run_a.py` 与 `patch_a_*.py` 保持原样，属于先前作者流程的兼容适配，
本次只记录哈希，不加载、不修改作者副本。旧原生 OBJ 的约 9.252 倍尺度不用于 A3。

NumPy/SciPy 实际安装包的完整许可（包括 OpenBLAS、LAPACK、GCC runtime、
libquadmath 等）保留在 `third_party_notices_a3/numpy.txt` 和 `scipy.txt`。
Python、其发行通知和可选 Pillow 通知也在该目录。未修改这些依赖。
本次不提交这些依赖的二进制、虚拟环境或源模型；保留通知不等于完成未来打包审查。

作者固定 `LICENSE.md` 的原始许可证包含 Tencent AI Lab、Christopher Hesse、
Charlie Nash 等署名及 Apache/CelebAMask-HQ 条款；保存为 `hifi3dface.txt`。
此通知的保留不把相应权重授权扩展到 A3，也不更改它们的研究限制。

真实原始数据由用户指定供本地处理。手工 ROI/点、相机外参、网格、预览图及
逐顶点支持度留在独立运行目录。无数据发布授权时，不提交或外传这些文件。
