# 多视角验证记录

## v0.9.1

- 修复输入来自真实六组采集的失败记录`output/mv_20261029_152628_711270`。原闭口左侧错误`neck=0.0274m`可复现：面部ICP RMSE为2.307mm，旧颈部单向全区域最近邻中位数27.375mm；旧下采样颈部点数正面344、左侧8119，存在大量非共同观察区域。
- 改为双向投影共同语义区域、可靠朝向和点到面比较，不按残差大小预筛；10mm阈值保留。闭口左侧修正后1.446mm，张口左/右1.607/1.288mm；闭口右侧共同颈部区域不足，显式未验证并警告，未冒充通过。面部四侧ICP RMSE分别2.307/3.103/2.219/2.720mm。这些是配准/一致性诊断，不能作为最终重建精度。
- `tools/check_multiview_registration.py --session output/mv_20261029_152628_711270/prepared/session.json --output output/validation_v091`：退出0。四侧面部配准通过，报告、逐视角外参及叠加位于`output/validation_v091/registration_check_20261029_153134_597128`。检查时为发布前工作树，版本号尚未写入0.9.1。
- `python -X utf8 -m unittest discover -s tests -v`：34项通过，退出0。新增不同可见范围造成的误报回归、朝向/远离相机25mm变化仍拒绝、缺失和超出视野明确未验证；扩充表情/颈部检查失败后的变换和诊断/叠加保存检查。合成几何仅验证软件行为。
- `tools/replay_multiview.py --session output/mv_20261029_152628_711270/prepared/session.json --output output/validation_v091_replay --quick --stage2-only`：退出0，真实六视角完成正面初始化、四侧配准、联合拟合及第二阶段网格/指标。结果`output/validation_v091_replay/mv_20261029_153308_864046`；500点及缩短调度只验证流程，不用于精度或收敛评估。
- `tools/check_multiview_outputs.py --run output/validation_v091_replay/mv_20261029_153308_864046 --output output/validation_v091`：退出0，保存后共享参数、状态正面坐标/外参、指标阶段检查通过。记录`output/validation_v091/mv_output_checks_20261029_153528_697881/checks.json`。此回归关闭第三阶段，没有对新采集数据执行顶点位移及几何保护检查。
- 编译及差异检查退出0。原始失败会话与采集输出不修改、不提交。未重新执行默认完整调度或六视角细化，缺少独立斜侧留出数据和人工核对轮廓，真实侧面重建精度仍未验收。

## v0.9.0

- `python -X utf8 -m unittest discover -s tests -v`：30项通过，退出0。覆盖四/六视角实际FLAME资产的数值雅可比、参数共享、有效性/可见性、旋转浮点误差处理、采集时间与内参逐帧快照、旧排队帧隔离和原始短序列保存；额外检查语义有效方程数量不改变视角之间的质量比例、输入次序独立及单视角原有权重。
- 联合细化检查覆盖带平移的相机方程/导数映射、多视角输入次序独立性、视角点数归一化、每状态只调用一次联合求解、位移只应用一次及总量/单步/平滑/几何保护。已知相机和合成平面用于验证数学与软件行为，不代表真实侧面效果。
- `tools/replay_multiview.py --legacy-pair output/20261027_143701_413530 --output output/validation_v090_full --stage2-only`：退出0，使用真实历史正面RGB-D和默认完整LM调度（未使用`--quick`），完成初始化、外参优化、20/5→50/10→100/30联合拟合和第二阶段指标。结果`output/validation_v090_full/mv_20261029_143952_483304`；文件级检查退出0，记录`output/validation_v090_contracts/mv_output_checks_20261029_145348_189798/checks.json`。
- 完整新入口的缩短调度回归（`--quick`，包括第二阶段、指标及第三阶段）退出0，结果`output/validation_v090_final/mv_20261029_143351_063986`；它只用于流程兼容性检查。
- 在上述默认完整LM输出上运行`tools/refine_multiview.py --run output/validation_v090_full/mv_20261029_143952_483304 --output output/validation_v090_release`，退出0。结果`output/validation_v090_release/mv_refine_20261029_145348_083472`，闭口/张口各由5023细分为20018顶点，最大位移分别1.331024/1.097682mm；指标字节未变、位移只应用一次。保存后的文件级检查退出0，记录`output/validation_v090_contracts/mv_output_checks_20261029_150143_376452/checks.json`，总量、单步、末端平滑及边长/面积/法线/折角保护全部通过。这次实际输入每状态仅一个正面；最终语义多视角归一化修改保持单视角尺度，另由上述自动检查验证。
- 原`tools/check_v3_contracts.py`退出0，旧细化位移、额外平滑和几何保护检查继续通过。真实检测/分割及六组采集页的离屏前端检查退出0，截图位于`output/frontend_v090`并已目视检查；这不等于真实硬件六视角采集。
- 评估工具在真实正面拟合内观测上执行成功，退出0，记录`output/validation_v090_evaluation/mv_evaluation_20261029_142229_834465`。缺少独立视角和人工核对轮廓，报告明确标为未完成真实验收；拟合内表面残差不能作为精度提高证据。
- 编译及差异检查退出0。过程中发现细分表面的射线查询临时内存过大，已改为分批精确查询；发现单精度旋转组合的数值非正交，已在容差内投影回SO(3)，状态正面外参显式为单位变换。没有增加自由尺度或放宽几何保护。

**未完成的真实验证：**真实移动相机采集、闭口/张口侧面配准与表情保持、侧脸/下颌角/前后厚度改善，以及未参与拟合的斜侧视角验收。需要新增真实侧面RGB-D、独立标定的留出视角和人工核对轮廓，分别报告正面/侧面表面误差、轮廓误差和有效覆盖率。所有现有真实回归只使用历史正面数据，未复制正面作为真实侧面证据。

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
