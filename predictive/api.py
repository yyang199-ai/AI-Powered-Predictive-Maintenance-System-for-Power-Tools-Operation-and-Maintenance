"""Local development API; external deployment requires identity/access control."""
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .store import Store
from .pipeline import load_bundle

app=FastAPI(title='电钻预测性维护研发 API',version='0.2.0',description='本地研发接口，尚未提供生产账号与设备权限隔离。')


def store():return Store(os.environ.get('MAINTENANCE_DB','artifacts/maintenance.sqlite3'))


class DeviceBinding(BaseModel):
    device_id:str=Field(min_length=1,max_length=100)
    name:str=Field(min_length=1,max_length=100)

class RepairRecord(BaseModel):
    component:str
    conclusion:str
    operation:str
    evidence:str

class Results(BaseModel):
    rows:list[dict]=Field(min_length=1,max_length=5000)


@app.get('/health')
def health():return {'status':'ok','mode':'local_research','data_source':'synthetic'}

@app.get('/devices')
def devices():return store().devices()

@app.post('/devices')
def bind(body:DeviceBinding):
    try:return {'device_id':store().bind(body.device_id,body.name)}
    except ValueError as error:raise HTTPException(422,str(error)) from error

@app.get('/devices/{device_id}/history')
def history(device_id:str,limit:int=1000):return store().history(device_id,max(1,min(limit,5000)))

@app.get('/events')
def events():return store().events()

@app.get('/events/{event_id}')
def event(event_id:str):
    for row in store().events():
        if row['id']==event_id:return row
    raise HTTPException(404,'事件不存在。')

@app.post('/events/{event_id}/acknowledge')
def acknowledge(event_id:str):
    db=store()
    if not any(e['id']==event_id for e in db.events()):raise HTTPException(404,'事件不存在。')
    db.acknowledge(event_id);return {'status':'acknowledged','fault_cleared':False}

@app.post('/events/{event_id}/tickets')
def create_ticket(event_id:str):
    try:return {'ticket_id':store().create_ticket(event_id)}
    except ValueError as error:raise HTTPException(422,str(error)) from error

@app.get('/tickets')
def tickets():return store().tickets()

@app.post('/tickets/{ticket_id}/repair')
def repair(ticket_id:str,body:RepairRecord):
    try:return store().repair(ticket_id,**body.model_dump())
    except ValueError as error:raise HTTPException(422,str(error)) from error

@app.post('/tickets/{ticket_id}/retest')
def retest(ticket_id:str,body:Results):
    # Only stored inference results are admissible; client risk scores are not trusted.
    db=store()
    keys=[(r.get('device_id'),r.get('run_id'),r.get('sequence')) for r in body.rows]
    trusted=[]
    for device,run,sequence in keys:
        record=db.db.execute('SELECT payload FROM telemetry WHERE device_id=? AND run_id=? AND sequence=?',(device,run,sequence)).fetchone()
        if record is None:raise HTTPException(422,'复测结果必须先由本地推理服务写入。')
        import json
        trusted.append(json.loads(record['payload']))
    try:return {'passed':db.retest(ticket_id,trusted)}
    except (ValueError,KeyError,TypeError) as error:raise HTTPException(422,str(error)) from error

@app.get('/models')
def models():
    if not Path('artifacts/model/manifest.json').exists():return []
    _,_,manifest=load_bundle();return [manifest]


@app.post('/telemetry')
def telemetry(body:Results):
    import pandas as pd
    from .data import FEATURES, QUALITY, CONTEXT
    from .pipeline import predict_trajectory
    from .alarms import annotate_alarms
    db=store();known={device['id'] for device in db.devices()}
    incoming=pd.DataFrame(body.rows)
    required=set(FEATURES+QUALITY+CONTEXT+['device_id','run_id','sequence','timestamp','mode'])
    if not required.issubset(incoming.columns):raise HTTPException(422,'缺少窗口字段：'+', '.join(sorted(required-set(incoming.columns))))
    if not set(incoming.device_id).issubset(known):raise HTTPException(422,'请先绑定设备。')
    try:
        incoming['timestamp']=pd.to_datetime(incoming.timestamp,utc=True,errors='raise').map(lambda t:t.isoformat())
        previous=[]
        for device in incoming.device_id.unique():previous.extend(db.history(device,5000))
        combined=pd.concat([pd.DataFrame(previous),incoming],ignore_index=True).drop_duplicates(['device_id','run_id','sequence'],keep='first')
        combined=combined.sort_values(['device_id','run_id','sequence']).reset_index(drop=True)
        predicted=annotate_alarms(predict_trajectory(combined))
        count=db.ingest(predicted.to_dict('records'))
        return {'inserted':count,'duplicates_ignored':len(body.rows)-count,'model_version':predicted.model_version.iloc[0]}
    except (ValueError,TypeError,KeyError) as error:raise HTTPException(422,str(error)) from error
