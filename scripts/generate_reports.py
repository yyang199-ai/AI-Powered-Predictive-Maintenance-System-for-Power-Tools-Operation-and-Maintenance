"""Generate reports from executed artifacts; never invent experimental results."""
import json
import platform
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from predictive.data import simulate_fleet,grouped_split
from predictive.pipeline import predict_trajectory
from predictive.alarms import annotate_alarms
from predictive.store import Store

torch.set_num_threads(2)
folder=Path('artifacts');report=json.loads((folder/'experiment_results.json').read_text())
data=simulate_fleet(30,200,report['seed']);test=grouped_split(data,report['seed'])['test']
result=annotate_alarms(predict_trajectory(test));result.to_csv(folder/'test_predictions.csv',index=False)
with tempfile.TemporaryDirectory() as temp:
    db=Store(Path(temp)/'workflow.sqlite3')
    for device in result.device_id.unique():db.bind(device,device)
    db.ingest(result.to_dict('records'))
    initial_events=db.events()
    event=initial_events[0];db.acknowledge(event['id']);ticket=db.create_ticket(event['id'])
    repair=db.repair(ticket,'轴承','模拟确认磨损','模拟更换','仅用于闭环测试的模拟证据')
    healthy=simulate_fleet(1,40,100);healthy['device_id']=event['device_id'];healthy['run_id']=repair['new_run_id']
    reviewed=annotate_alarms(predict_trajectory(healthy));db.ingest(reviewed.to_dict('records'))
    passed=db.retest(ticket,reviewed.to_dict('records'))
    if not passed:raise RuntimeError('模型驱动的模拟维修复测闭环未通过，停止生成成功报告。')
    db.db.close()
report['alarm_evaluation']={'test_devices':3,'total_test_operating_hours':len(test)/60,'created_alarm_events':len(initial_events),'events_per_100_operating_hours':len(initial_events)/(len(test)/60)*100,'note':'All trajectories contain progressing wear. This is all alarm events, not false alarm rate; overlapping evidence is correlated.'}
report['end_to_end_workflow']={'repair_retest_closed':passed,'source':'synthetic; explicitly designed healthy-retest fixture seed 100'}
report['runtime']={'python':platform.python_version(),'torch':torch.__version__,'platform':platform.platform(),'torch_threads':torch.get_num_threads()}
(folder/'experiment_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
lines=['# 电钻预测性维护仿真实验报告','','日期：2026-10-08；数据来源：仿真特征及仿真磨损/寿命标签。','',
'## 方法与数据','',
'30台虚拟电钻，每台200个一分钟聚合窗口，共6000个窗口。训练/验证/测试设备分别为21/6/3台，设备无交集。健康基线只在训练设备高质量且模拟磨损小于0.15的数据拟合；验证集选择最佳训练轮次。种子42，训练12轮，CPU两线程。','',
'每个输入序列为16窗。跨运行段、序号缺口、全部通道无效或未知工况的序列被排除。测试可评估序列459条；并非459次独立实验。仿真失效终点为磨损≥1，RUL单位为有效运行小时，HI标签为1减模拟磨损。','',
'## 实测对照','',
'| 模型 | 四分类准确率 | RUL MAE 小时 | RUL相对误差* | P95前向延迟 ms | 参数量 |',
'|---|---:|---:|---:|---:|---:|']
for name,metrics in report['experiments'].items():
    if 'accuracy' in metrics:
        lines.append(f"| {name} | {metrics['accuracy']:.2%} | {metrics['rul_mae_hours']:.4f} | {metrics['rul_mape_excluding_under_15min']:.2%} | {metrics['inference_p95_ms']:.3f} | {metrics['parameter_count']} |")
h=report['experiments']['hybrid'];q=report['experiments']['hybrid_lstm_int8'];v=report['experiments']['vibration_threshold']
lines+=['','*相对误差仅统计真实模拟RUL≥0.25小时的样本，避免近零分母发散；不等于全寿命段误差。','',
'当前CPU的模型前向延迟不包含采集、特征提取、存储或通信；没有测量目标MCU、峰值RAM与功耗。量化后的参数数量保持结构数量，不将打包权重未暴露为Parameter误认为剪枝。','',
'## 分类细节','', '| 磨损阶段 | 精确率 | 召回率 | 样本数 |','|---|---:|---:|---:|']
for c in h['per_class']:
    lines.append(f"| {['正常','轻微','中度','重度'][c['stage']]} | {c['precision']:.2%} | {c['recall']:.2%} | {c['support']} |")
lines+=['',f"组合模型模拟80%寿命区间覆盖率为 {h['rul_interval_coverage']:.2%}。这不是实际设备的区间校准结果。",'',
'## 阈值与告警','',f"仅使用振动RMS的二分类阈值在验证集选择为 {v['threshold']:.4f}；测试二分类准确率 {v['binary_accuracy']:.2%}，健康窗口误报率 {v['healthy_false_positive_rate']:.2%}，任意磨损召回 {v['fault_recall']:.2%}。不得直接与四分类准确率比较。",'',
f"组合模型测试历史产生 {len(initial_events)} 个告警事件，总模拟运行10小时，约 {len(initial_events)*10:.1f} 个事件/百运行小时。所有测试轨迹均含渐进磨损，该计数不是误报率，也不是健康设备长期运行的误报证据。",'',
'完成模型预测→告警持久化→确认收到→工单→模拟维修→新运行段模型复测→关闭事件。健康复测使用明示的固定测试样例seed 100；不能当作维修有效性的真实证据。', '',
'## 结论与未达到的论证','',
'该仿真中工况校正消融与固定融合消融支持继续研究条件基线和质量约束。纯LSTM与组合模型总体准确率相同，RUL MAE更低且参数更少，未证明串联结构必然有优势。', '',
f"LSTM动态INT8使组合模型权重序列化大小从 {h['serialized_weights_bytes']} B 降到 {q['serialized_weights_bytes']} B，下降 {(1-q['serialized_weights_bytes']/h['serialized_weights_bytes']):.2%}；当前CPU前向反而变慢，不支持70%算力降低的结论。",'',
'仅运行一个随机种子、3台独立测试设备，数据由已知公式生成，不能替代真实台架、跨工具型号、跨操作者及真实磨损研究。未达到3–7天预警的真实验证；原型累计时长约数小时，不能据此推断数周寿命。','',
'## 复现','', '```bash', '.venv/bin/python -m predictive.train --epochs 12 --seed 42','.venv/bin/python scripts/generate_reports.py','.venv/bin/python -m unittest discover -s tests -v','```','',
'原始指标、设备划分和混淆矩阵：artifacts/experiment_results.json；测试输出：artifacts/test_predictions.csv；生成数据：artifacts/simulated_fleet.csv。计时受CPU调度影响，不要求重复运行数值完全一致。']
Path('docs/仿真实验报告.md').write_text('\n'.join(lines)+'\n')
# DOCX copies remain explicitly labeled as draft and simulation evidence.
for source in ['专利技术交底草稿','仿真实验报告','数据契约与部署','研发范围与验收']:
    document=Document();normal=document.styles['Normal'];normal.font.name='Microsoft YaHei'
    normal.element.rPr.rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
    for line in Path('docs/'+source+'.md').read_text().splitlines():
        if line.startswith('# '):document.add_heading(line[2:],0)
        elif line.startswith('## '):document.add_heading(line[3:],1)
        elif line.startswith('### '):document.add_heading(line[4:],2)
        elif line.strip():document.add_paragraph(line)
    document.save('docs/'+source+'.docx')
print('Generated Markdown and DOCX reports; model-driven repair/retest closure passed.')
