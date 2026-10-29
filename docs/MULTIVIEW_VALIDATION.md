# 多视角验证记录

## v0.8.0

- 完整合同18项通过（退出0）：保留v0.7检查，新增六相机实际FLAME资产上的完整残差/顶点/标点数值雅可比、六步骤采集、两状态共享参数、张口变化/缺少唇点拒绝，以及指标绑定正面观测/闭口ArUco的入口检查。
- 六组采集流程中的检测/相机帧与指标来源隔离检查使用测试替身；未称作真实六视角采集。
- `tools/replay_multiview.py --legacy-pair output/20261027_143701_413530 --output output/validation_v080 --quick`：用于兼容性烟雾检查；采用真实正面输入和缩短调度，不作为收敛精度评估。
- 编译和差异检查退出0。真实侧面、张口保持及独立斜侧效果仍需真实数据验收。

## v0.7.0

- `python -X utf8 -m unittest discover -s tests -v`：11项通过。真实FLAME资产上的四相机顶点/标点/完整残差数值雅可比、共享正则、无缩放坐标/投影、遮挡、侧面缺点掩码、同视角融合与四步骤采集状态机（检测和时间为测试替身）。
- `tools/replay_multiview.py --legacy-pair output/20261027_143701_413530 --output output/validation_v070`：退出0；默认完整调度，原正面初始化、仅外参阶段、三档联合LM及指标完成，178.40秒。结果`output/validation_v070/mv_20261029_135430_419641`，两状态stage2各5023顶点；未生成stage3。
- `tools/check_frontend.py --capture output/20261027_143701_413530/capture_pair.npz`：退出0；真实检测、TUFA、分割和离屏采集页通过。每帧68标点，点云123115/130956点。结果`output/frontend_validation`，截图已目视检查。
- `tools/check_v3_contracts.py`：退出0，原位移/平滑/几何保护合同通过（合成几何）。生成记录归档于`output/legacy_contracts_v070`。
- `tools/check_jaw_pipeline.py --run output/20261027_143701_413530`：退出0；原GUI工作线程真实执行ArUco、stage1/2、细分、stage3和保存，缓存分割/标点、测试替身渲染。生成结果归档`output/legacy_pipeline_v070`，原检查记录恢复。
- 编译及`git diff --check`通过。环境Python3.10、torch2.0.1，CUDA可用；ONNX实际回退CPU。
- 过程发现并修正：旧前端检查缺少参数；采集测试连续注入图片的相同时间戳导致警告等待，测试改为明确30ms间隔，生产时间检查未放宽。

**未验证：**真实硬件移动相机采集、真实侧面ICP/联合拟合改善、独立斜侧验收。现有输入只有正面，不能得出侧面/下颌角/厚度精度结论。四视角导数合同使用合成相机与真实FLAME资产，不作为真实效果证据。
