# 多视角 RGB-D（v0.8.0）

实际调用链：`MultiCaptureWidget → MultiPipelineWorker → run_multiview → fit_multiview → MultiProblem`。
原双帧 `CaptureWidget → PipelineWorker → run_pair` 路径保留。

## 采集

运行 `start.cmd`，默认选择“闭口三视角 + 张口三视角”，依次采集闭口正面、左约40°、右约40°，再张口正面、左约40°、右约40°。
可切回四组（闭口三视角+张口正面）或原双帧模式。张口三次必须保持同样开口程度和头颈姿态。
张口侧面至少需要两组有效唇距，并检查配准后唇点位置；不足或不一致时提示重采。
角度只是引导。人保持头颈姿态及同状态表情，移动相机；每个位置停稳后点击采集。
每次融合该视角最近五帧有效深度中位数，记录时间、实际内参、深度抖动与质量权重。
切换视角或警告后清空短序列，禁止跨视角融合。正面闭口需ArUco，侧面不需要。
不支持转头后把颈部当作随头刚性运动；声明为head_turned的数据拒绝。
运动过大、正面标点无效、稳定重叠不足、配准失败、开口程度变化或头颈不一致时提示重采。
自动检查不能证明人完全不动，仍需遵守采集操作。
`output/capture_mv_*/attempt_*/`保留每次原始尝试，`complete/session.json`是完整会话。
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
视角数据项按有效权重归一后按质量加权；shape正则一次，expression每状态一次。

## 指标与验收

指标仍基于stage2。`report.json.metric_sources`指明两种状态正面及闭口ArUco来源。
网格、标点与ArUco在对应正面相机系，侧面标点不混入指标。原指标公式不变。
合成几何仅验证数学实现；已有真实正面数据仅验证离线可执行性。
缺少真实侧面及独立斜侧数据，**真实侧面改善尚未验证**。
真实验收应固定同组数据与stage2调度，比较仅正面和多视角；用未参与拟合的斜侧观测，分别报告正面/侧面表面距离、轮廓法向误差及有效覆盖率。
总损失下降、合成或正面复制数据不能代替真实验收。
