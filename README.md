# 电钻 AI 预测性维护研发原型

对应用户研发任务书及端到端技术路线。第一版：电钻轴承磨损，多工况仿真、质量融合、Transformer–LSTM、告警、维修工单与复测闭环。运行在 Python 3.12 CPU 和 Streamlit 网页，无需真实数据、GPU或密钥。

**所有模型效果均来自仿真。** 未完成真实硬件采集、MCU部署、原生手机APP、正式专利查新与提交。HI不是剩余寿命百分比；RUL单位为有效运行小时，不直接换算为自然日。

## 安装、训练、运行

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m predictive.train --epochs 12 --seed 42
.venv/bin/python scripts/generate_reports.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

CPU版 PyTorch 由锁定文件中的官方CPU索引下载，保持TLS验证。训练产物在 `artifacts/model`，仿真数据与结果报告在 `artifacts`；仓库包含已训练模型及仿真报告，其他本地数据库与生成ZIP被Git忽略。可通过上面命令重建训练产物。已经训练且依赖未变化时，无需每次重新训练。

可选本地 API：

```bash
.venv/bin/python -m uvicorn predictive.api:app --host 127.0.0.1 --port 8000
```

网页直接使用本地SQLite，无需API进程。API供后续硬件或APP联调，当前没有生产身份权限，不应直接作为公网服务部署。

## 页面使用顺序

1. 选择模拟数据和设备，查看历史工况、健康指数、数据质量及条件寿命区间。
2. 点击“绑定仿真设备并保存分析记录”，产生可追溯窗口和合并告警；重复点击不重复存储。
3. 在告警页确认收到、创建工单；填写现场结论、维修操作和明确注明模拟的证据。
4. 维修后生成新运行段；执行模拟复测。只有同设备、新运行段、连续5个有效低风险且稳定工况结果才关闭事件。
5. 查看实验与模型页，下载CSV或实验原始报告。CSV格式已从原来的三列演示升级为工况及质量窗口格式，页面可下载示例。

仿真健康复测是专门配置的开发测试样例，不代表真实修理成功。上传数据不会自动获得真实设备校准：当前模型仍使用模拟健康基线。

## 交付物

- `docs/仿真实验报告.md` / `.docx`：实际运行的对照指标及局限。
- `docs/专利技术交底草稿.md` / `.docx`：技术方案、实施例、拟保护组合及待查新问题。
- `docs/数据契约与部署.md` / `.docx`：单位、接口、质量、状态和部署边界。
- `docs/研发范围与验收.md` / `.docx`：任务目标与证据对照。
- `docs/系统架构.md`：系统和工单流程图。
- `artifacts/experiment_results.json`：设备划分、各类召回、混淆矩阵、量化和CPU计时。
- `predictive/`：仿真、信号辅助函数、基线、时序模型、告警、持久化及API。
- `tests/`：界面、模型、数据边界、去重、维修闭环和接口测试。

旧的 `maintenance.py` 保留为最初IsolationForest演示及回归参考，当前网页使用 `predictive/` 新流程。

云环境发布、GitHub上传和公网部署是三个独立操作。

## Windows 快速查看

在GitHub点击 **Code → Download ZIP**，右键ZIP选择“全部解压”。安装Python 3.12后，打开含 `app.py` 的文件夹，在地址栏输入 `cmd` 回车，逐行执行：

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

安装完成后浏览器通常自动打开；也可手动输入 `http://localhost:8501`。保持命令窗口打开。仓库已包含训练好的模型，首次查看无需重新训练。当前验证平台为云端Linux/Python 3.12，Windows运行需在你的电脑上确认；安装报错时保留完整错误信息以便定位。
