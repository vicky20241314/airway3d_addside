# 整体头部组合：本次研究使用边界

这是对原 A3 无模型入口的后续实验，**组合结果含 FLAME 模型先验**。
不能把原 A3 的“未加载模型”说明外推成组合网格不受模型条件约束。

2026-10-30，用户明确确认：已接受现有 FLAME 的官方学术许可，此次仅本地
非商业研究。原始声明记录在仓库外 `head-research-authorization.json`，绑定
现有模型 SHA-256 `8fb1af0db1abb51053ead8fd1f2624a63d01c9602f4a4fb4ea23bd2c82017fa0`。
它与 `docs/asset_manifest.json`、实际资产哈希及既有 B 报告记录吻合。
这是用户报告的许可接受情况，未独立认证原始下载合同，不代表取得商业授权。

| 组件 | 来源与固定版本 | 本次允许用途和分发处理 | 证据与未明确事项 |
|---|---|---|---|
| FLAME 派生整体头部 | 旧项目模型哈希见上；B 的 `parameters.npz` 或基线 stage2 OBJ；基线 commit `3e409af77a4686b0cf435fff2230b90a8d26e11d` | 依据用户确认的学术许可做本地非商业研究；原模型不修改、不加载、不复制、不发布；派生网格留本地 | 用户声明、资产清单、B 模型哈希、官方许可；未证明为 FLAME 2023 Open，商业/第三方分发范围未授予 |
| 旧 B / 原项目基线算法结果 | 研究提交 `159a894...` 已产生的真实采集结果；每次输入文件 SHA 在本地侧车 | 作为已存在的头部先验读入，维持原资产、上游数据及结果条件；不重新运行 TUFA、分割或权重 | 上游完整训练权利链未因此清理，不为商业和再分发出具结论 |
| 新数值适配代码 | `head_adapter.py`、`run_head_combined.py`、测试及摘要工具，本任务新增 | 可审阅的独立代码入口，不复制 FLAME/TUFA 算法源码，不替换默认入口 | 与作者 A1、B 及旧项目代码明确区分；不是论文复现或原作者认可的产品 |
| NumPy / SciPy / Python | 只读 env-b：1.26.4 / 1.11.4 / 3.10.22；原许可证及二进制哈希已归档 | 原运行依赖许可及传递组件条件继续适用；环境、DLL 不随本次分发 | `A_DEPENDENCIES.json`、`a3-license-evidence.json`、`third_party_notices_a3/`；供应商合同及未来二进制发布另审 |
| Pillow | 11.1.0，仅本机灰模 QA | 不打包、不分发二进制；无纹理或外部渲染服务 | 原始 MIT-CMU 通知已保留；未来编解码二进制发布另审 |
| 真实数据及 A3 几何 | 原来同一单相机采集的真实帧；A3 19 个接纳帧，两个明确参考状态 | 用户授权本地处理；RGB、深度、点、姿态、网格、图和支持度留本地 | 没有公开传播或商业数据主体授权；输入 SHA/真实参考在私有完整报告 |

模型、参数、网格的转换和导出没有解除许可限制。本次不联系作者，不代表用户
作新承诺，也没有因包装代码 MIT 或 README 的开放说明而将现有资产认定为 Open 版。
学术 FLAME 与 FLAME 2023 Open 的区别以[官方许可](https://flame.is.tue.mpg.de/modellicense.html)
为证据；本次按用户已接受的学术用途范围执行，不作为产品商业/分发审查通过。

FLAME 署名：Tianye Li, Timo Bolkart, Michael J. Black, Hao Li, Javier Romero,
“Learning a model of facial shape and expression from 4D scans,” ACM TOG,
SIGGRAPH Asia 2017，[论文 DOI](https://doi.org/10.1145/3130800.3130813)。
既有 B 是受 FaceCap 思想启发的独立试验，不是完整 FaceCap 复现。
既有 A3 是参考 RGB-D 多视角思想的独立观测表面实现，不是 Tencent 方法复现。

本次 Git 仅发布新代码、必要说明和聚合数值摘要，不发布 FLAME、头部模板、
身份/表情参数、真实数据、派生网格、相机变换或预览图。
