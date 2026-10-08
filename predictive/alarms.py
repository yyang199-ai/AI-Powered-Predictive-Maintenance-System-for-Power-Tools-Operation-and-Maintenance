"""Per-run degradation confirmation; no sensor data is not a healthy result."""
from collections import deque
import math


class AlarmState:
    def __init__(self):
        self.evidence = deque(maxlen=5)
        self.state = '正常'
        self.run_id = None
        self.previous_load = None
        self.previous_sequence = None
        self.recovery = 0

    def update(self, row):
        run_id = row.get('run_id')
        sequence = int(row['sequence'])
        if self.run_id != run_id or (self.previous_sequence is not None and sequence != self.previous_sequence+1):
            self.evidence.clear();self.recovery=0
        self.run_id, self.previous_sequence = run_id, sequence
        load = float(row['load'])
        switching = self.previous_load is not None and abs(load-self.previous_load)>0.15
        self.previous_load = load
        validity = row.get('validity')
        risk = row.get('risk_score')
        if validity not in ('valid','degraded') or risk is None or not math.isfinite(float(risk)):
            self.evidence.clear();self.recovery=0
            return {'state':'数据不足','reason':validity or 'missing_prediction','confirmed':False}
        if switching:
            self.evidence.clear();self.recovery=0
            return {'state':self.state,'reason':'工况切换，暂停退化累计','confirmed':False}
        self.evidence.append(float(risk))
        high = sum(value >= 0.8 for value in self.evidence)>=3
        warning = sum(value >= 0.4 for value in self.evidence)>=3
        if high:
            self.state='高风险'; self.recovery=0
        elif warning:
            self.state='预警'; self.recovery=0
        elif float(risk)<0.25:
            self.recovery+=1
            if self.recovery>=5:self.state='正常'
        else:self.recovery=0
        return {'state':self.state,'reason':'最近5个有效窗口至少3个越阈值；恢复需连续5个低风险窗口','confirmed':high or warning}


def annotate_alarms(frame):
    result=frame.copy();states={};annotations=[]
    for row in result.to_dict('records'):
        machine=states.setdefault(row['device_id'],AlarmState())
        annotations.append(machine.update(row))
    result['health_state']=[a['state'] for a in annotations]
    result['alarm_reason']=[a['reason'] for a in annotations]
    result['alarm_confirmed']=[a['confirmed'] for a in annotations]
    return result
