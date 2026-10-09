# 电动工具实验与诊断平台

当前版本完成任务书第二阶段的**电脑模拟实验台与公开轴承数据算法预验证**。默认网页使用真实 CWRU 振动，附带已预处理数据和 CPU 模型；另提供三类电动工具虚拟实验台与 NASA IMS 完整退化趋势。

[![在 GitHub Codespaces 打开](https://github.com/codespaces/badge.svg)](https://codespaces.new/yyang199-ai/AI-Powered-Predictive-Maintenance-System-for-Power-Tools-Operation-and-Maintenance?quickstart=1)

**先读 [早上先看这里](docs/phase2/早上先看这里.md)。** 当前是研究 Demo：主协议留出记录测试准确率约 81.38%，正常记录误报明显，未达到任务书 92% 目标。尚无真实工具采集、已验证的真实寿命模型、MCU 部署或正式专利查新结论。

## 在 GitHub 浏览器里运行

点击上面的 **Open in GitHub Codespaces**，登录 GitHub 并创建个人 Codespace。首次等待依赖自动安装；配置会自动启动实验平台，并打开端口 8501 的网页。不需要本地安装 Python 或 VS Code。

若网页没有自动弹出，在浏览器编辑器底部打开 **Ports（端口）**，找到 **8501**，点击 **Open in Browser（在浏览器中打开）**。需要手动启动时，在终端执行 `bash scripts/start_codespaces.sh`。端口默认使用 Codespaces 私有转发，需要你的 GitHub 登录；停止 Codespace 后网页停止，重新启动后服务自动恢复。

Codespaces 是个人云工作区，使用资格和额度以 GitHub 提示为准。这里提供了启动配置，尚未代你创建 Codespace，也没有发布永久公网网站。

## Windows：直接打开

1. GitHub **Code → Download ZIP**，右键 ZIP 选择“全部解压”；也可在 VS Code 克隆仓库。
2. 安装 **Python 3.12**，包括 Python Launcher。
3. 进入含 `app.py` 的文件夹，双击 **启动实验平台.bat**。首次安装依赖并校验数据和模型，以后直接启动。
4. 浏览器输入 `http://localhost:8501`。保持命令窗口打开；停止时按 Ctrl+C。

首次查看**无需重新训练或另下载 CWRU**。已在 Linux／Python 3.12 CPU 验证；Windows 脚本已提供，尚未在你的电脑实测。手动启动可在项目文件夹执行：

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.venv\Scripts\python.exe scripts\verify_phase2.py
.venv\Scripts\python.exe -m streamlit run app.py
```

## 网页分别做什么

| 左侧页面 | 操作 | 结果含义 |
|---|---|---|
| 公开轴承预验证 | 选测试记录、分析窗口、对照真实标签、看波形频谱 | 四类故障部位预测；人工缺陷尺寸来自已知标签，不是自然磨损预测 |
| 虚拟电动工具实验台 | 选三种工具、四档损伤、负载转速；生成下载三路波形 | 损伤是生成条件，用于验证流程和设计实验 |
| 模型与使用说明 | 看模型对照、使用说明、NASA 退化趋势 | 区分实际分类、末端代理标签与未完成的真实验证 |

旧版告警、工单、仿真 HI／RUL 和维修复测保留在 `legacy_app.py`：`python -m streamlit run legacy_app.py --server.port 8502`。它的模型仍只来自仿真，不能用于公开波形或真实工具。

## 已准备的成果

- CWRU：**40 个真实 MAT → 9346 窗**。28／8／4 完整记录分区，窗口 6625／1894／827；小波降噪、抗混叠重采样、FFT／时域／包络特征和标签已完成。40 个 MAT 与官网独立下载字节校验一致；正常数据 48 kHz 的解释依据单独记录，MAT 无采样率字段。
- NASA IMS run 2：**984 次快照 × 4 轴承 = 3936 行特征**。云工作区保留完整原始波形；仓库附带特征、趋势和校验清单。一个已描述失效事件，未训练可信寿命模型。
- 虚拟台：**48 组受控实验**，三类工具×四档损伤×两负载×两种子，明确来源与单位。
- 模型：纯 CNN／标准注意力／线性注意力公平比较，按验证集选择；另有完整负载 3 留出验证。详见 [模型实验报告](docs/phase2/公开轴承模型实验报告.md)。

| 文件 | 用途 |
|---|---|
| `artifacts/bearing/data/dataset.npz` | 处理好的 X、y、split、record_id 与工况／尺寸标签 |
| `artifacts/bearing/data/features.csv` / `records.csv` | Excel 可打开的窗口特征与原始记录身份 |
| `artifacts/bearing/model.pt` / `manifest.json` | 默认分类器、实测结果、来源与校验 |
| `artifacts/bearing/load3_holdout/manifest.json` | 跨负载留出实验 |
| `data/public/NASA_IMS/prepared/features.csv` | 完整退化趋势，严格标明末端代理标签 |
| `artifacts/lab/runs.csv` | 48 组虚拟多传感器对照 |

缺少的电流、温度、自然磨损与真实寿命标签不由模拟值补入。大原始下载留在云工作区并被 Git 忽略，可用脚本重建。

## Linux／云工作区

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python scripts/verify_phase2.py
.venv/bin/python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

服务运行在工作区；保存云环境、上传 GitHub 与公网部署是不同操作。

## 重建数据、训练与验证

```bash
# 原始下载与预处理；已有 raw 可加 --offline
python scripts/prepare_public_data.py
# 三候选训练比较；打开网页不用执行
python scripts/train_bearing.py --epochs 15
# 跨负载留出，另存以免覆盖默认模型
python scripts/train_bearing.py --epochs 15 --protocol load3_holdout --output artifacts/bearing/load3_holdout
# NASA 完整 run2 下载与处理参数
python scripts/prepare_nasa_ims.py --help
# 48组虚拟对照；加 --save-waveforms 保存全部波形
python scripts/run_virtual_lab.py --output artifacts/lab --duration 2 --seeds 2
python -m unittest discover -s tests -v
```

说明：[模型结构与计算过程](docs/phase2/模型结构与计算过程.md)、[网页使用指南](docs/phase2/网页使用指南.md)、[虚拟实验台设计](docs/phase2/虚拟实验台设计.md)、[公开数据说明](docs/phase2/公开数据说明.md)、[第二阶段验收与创新边界](docs/phase2/第二阶段验收与创新边界.md)。早期专利交底、查新与汇报材料在 `docs/`，旧仿真指标不能当作本次真实公开数据指标。

可选原仿真 API：`python -m uvicorn predictive.api:app --host 127.0.0.1 --port 8000`。新版网页直接调用分类器，无需 API 服务；此 API 没有生产身份权限。
