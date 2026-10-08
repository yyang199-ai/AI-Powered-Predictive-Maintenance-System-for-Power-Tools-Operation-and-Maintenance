from pathlib import Path
import json
import pandas as pd
import streamlit as st
import torch
from predictive.data import simulate_fleet, FEATURES, QUALITY, CONTEXT
from predictive.pipeline import predict_trajectory
from predictive.alarms import annotate_alarms
from predictive.store import Store

torch.set_num_threads(2)
st.set_page_config(page_title="电钻预测性维护研发系统",layout="wide")
st.title("电钻预测性维护研发系统")
st.caption("仿真研究原型 · 工况校正 / 质量融合 / Transformer–LSTM / 告警与维修复测")
st.info("当前模型仅在模拟特征与模拟寿命标签上训练。健康指数不是剩余寿命百分比；寿命区间单位为有效运行小时，真实设备效果待验证。")

@st.cache_data
def evaluate_csv(content):
    data=pd.read_csv(content)
    return annotate_alarms(predict_trajectory(data))

@st.cache_data
def evaluate_simulation(seed):
    return annotate_alarms(predict_trajectory(simulate_fleet(devices=4,windows=200,seed=seed)))

if not Path('artifacts/model/manifest.json').exists():
    st.error("模型尚未训练，请运行 .venv/bin/python -m predictive.train")
    st.stop()
source=st.sidebar.radio("数据来源",["模拟数据","上传 CSV"])
if source=="模拟数据":
    seed=int(st.sidebar.number_input("仿真随机种子",min_value=0,max_value=999999,value=123))
    with st.spinner("分析设备历史…"):
        result=evaluate_simulation(seed)
else:
    st.sidebar.write("上传包含设备身份、工况、质量分数及六维窗口特征的 CSV；格式见数据契约。")
    upload=st.sidebar.file_uploader("窗口特征 CSV",type='csv')
    if upload is None:
        st.info("上传 CSV 后分析，也可以切换到模拟数据。")
        st.stop()
    try:
        data=pd.read_csv(upload)
        required=set(FEATURES+QUALITY+CONTEXT+['device_id','run_id','sequence','timestamp','mode'])
        missing=required-set(data.columns)
        if missing:raise ValueError('缺少列：'+', '.join(sorted(missing)))
        if data.duplicated(['device_id','run_id','sequence']).any():raise ValueError('窗口身份重复。')
        data['timestamp']=pd.to_datetime(data.timestamp,utc=True,errors='raise').map(lambda t:t.isoformat())
        data=data.sort_values(['device_id','run_id','sequence']).reset_index(drop=True)
        result=annotate_alarms(predict_trajectory(data))
    except (ValueError,KeyError,UnicodeError,pd.errors.ParserError) as error:
        st.error(str(error));st.stop()

store=Store()
tabs=st.tabs(['设备与健康','告警与工单','实验与模型'])
with tabs[0]:
    device=st.selectbox('设备',sorted(result.device_id.unique()))
    selected=result[result.device_id==device]
    latest=selected.iloc[-1]
    a,b,c,d=st.columns(4)
    a.metric('历史窗口',len(selected))
    b.metric('最近评估状态',latest.health_state)
    c.metric('健康指数',f'{latest.health_index:.1%}' if pd.notna(latest.health_index) else '数据不足')
    d.metric('结果有效性',latest.validity)
    st.write(f"最后采样时间：{latest.timestamp}；数据来源：{latest.source}（历史回放，不代表在线连接）")
    if pd.notna(latest.rul_lower_hours):
        st.write(f"模拟剩余运行小时区间：{latest.rul_lower_hours:.2f}–{latest.rul_upper_hours:.2f}，中值 {latest.rul_median_hours:.2f}。条件：未来使用强度与仿真工况相近。")
    else:st.warning('当前不输出寿命区间：历史不足、未知工况或关键传感器缺失。')
    st.subheader('工况与传感器趋势')
    st.line_chart(selected.set_index('sequence')[['vibration_rms','current_rms','temperature_rise']])
    st.subheader('健康与数据质量')
    st.line_chart(selected.set_index('sequence')[['health_index']+QUALITY])
    st.dataframe(selected[['sequence','load','rpm','health_state','validity','risk_score','rul_median_hours']],use_container_width=True)
    if st.button('绑定仿真设备并保存分析记录'):
        for identity in result.device_id.unique():store.bind(identity,identity+' 电钻')
        count=store.ingest(result.to_dict('records'))
        st.success(f'新增 {count} 个窗口；重复记录自动忽略，持续告警合并为事件。')
    st.download_button('下载分析结果',result.to_csv(index=False).encode('utf-8-sig'),'maintenance_results.csv','text/csv')
    st.download_button('下载输入 CSV 示例',simulate_fleet(1,200).to_csv(index=False).encode('utf-8-sig'),'input_example.csv','text/csv')
with tabs[1]:
    st.write('先保存分析记录，再按告警 → 工单 → 维修 → 新运行段复测处理。确认收到只改变处理进度。')
    events=store.events()
    if not events:st.info('尚无持久化告警事件。')
    for event in events:
        with st.expander(f"{event['device_id']} · {event['severity']} · {event['status']} · {event['id'][:8]}"):
            st.json(json.loads(event['evidence']))
            if event['status']=='open' and st.button('确认收到',key='ack'+event['id']):
                store.acknowledge(event['id']);st.rerun()
            if event['status']!='closed' and st.button('创建或查看工单',key='ticket'+event['id']):
                st.success('工单编号：'+store.create_ticket(event['id']))
    for ticket in store.tickets():
        with st.expander(f"工单 {ticket['id'][:8]} · {ticket['status']}"):
            if ticket['status']=='open':
                with st.form('repair'+ticket['id']):
                    component=st.text_input('检查部件',value='轴承')
                    conclusion=st.selectbox('现场结论',['已确认磨损','未发现故障','预防性更换','传感器问题','原因不明'])
                    operation=st.text_input('维修操作')
                    evidence=st.text_area('证据与检查记录（仿真时请注明模拟）')
                    if st.form_submit_button('记录维修，进入待复测'):
                        try:
                            record=store.repair(ticket['id'],component,conclusion,operation,evidence)
                            st.success('新运行段：'+record['new_run_id']);st.rerun()
                        except ValueError as error:st.error(str(error))
            elif ticket['status']=='pending_retest':
                repair=json.loads(ticket['repair']);st.json(repair)
                st.write('复测要求：新运行段、相同设备、连续5个有效低风险窗口、稳定工况。维修标签仍需人工审核。')
                if st.button('执行模拟健康复测',key='retest'+ticket['id']):
                    event=next(e for e in events if e['id']==ticket['event_id'])
                    retest=simulate_fleet(1,40,seed=100)
                    retest['device_id']=event['device_id'];retest['run_id']=repair['new_run_id']
                    # New recording follows the event, retaining distinct sampling and receive times.
                    previous=pd.Timestamp(json.loads(event['evidence'])['timestamp'])
                    retest['timestamp']=[(previous+pd.Timedelta(minutes=201+i)).isoformat() for i in range(len(retest))]
                    reviewed=annotate_alarms(predict_trajectory(retest))
                    store.ingest(reviewed.to_dict('records'))
                    passed=store.retest(ticket['id'],reviewed.to_dict('records'))
                    if passed:st.success('模拟复测通过，事件已关闭。')
                    else:st.warning('复测未通过，工单保留待复测状态。')
            else:st.json(json.loads(ticket['retest']))
    st.subheader('操作审计')
    st.dataframe(pd.DataFrame(store.audit_history()),use_container_width=True)
with tabs[2]:
    manifest=json.loads(Path('artifacts/model/manifest.json').read_text())
    st.json(manifest)
    report_path=Path('artifacts/experiment_results.json')
    if report_path.exists():
        report=json.loads(report_path.read_text())
        rows=[]
        for name,metrics in report['experiments'].items():
            rows.append({'模型':name,'四分类准确率':metrics.get('accuracy'),'RUL MAE（小时）':metrics.get('rul_mae_hours'),'模型推理P95（ms）':metrics.get('inference_p95_ms'),'参数量':metrics.get('parameter_count')})
        st.dataframe(pd.DataFrame(rows),use_container_width=True)
        st.caption('阈值基线是二分类；神经网络为四分类，不直接比较两种准确率。所有结果来自仿真，CPU 延迟仅含模型前向。')
        st.download_button('下载实验原始报告',report_path.read_bytes(),'experiment_results.json','application/json')
