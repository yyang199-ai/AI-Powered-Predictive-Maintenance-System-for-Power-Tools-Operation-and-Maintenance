"""SQLite maintenance workflow, deduplication, and audit history."""
import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def now():return datetime.now(timezone.utc).isoformat()


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


class Store:
    def __init__(self,path='artifacts/maintenance.sqlite3'):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path,check_same_thread=False)
        self.db.row_factory=sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS devices(id TEXT PRIMARY KEY,name TEXT NOT NULL,model TEXT NOT NULL,created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS telemetry(device_id TEXT NOT NULL REFERENCES devices(id),run_id TEXT NOT NULL,sequence INTEGER NOT NULL,sampled_at TEXT NOT NULL,received_at TEXT NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(device_id,run_id,sequence));
        CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,device_id TEXT NOT NULL REFERENCES devices(id),run_id TEXT NOT NULL,severity TEXT NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL,evidence TEXT NOT NULL);
        CREATE UNIQUE INDEX IF NOT EXISTS active_event ON events(device_id,run_id) WHERE status!='closed';
        CREATE TABLE IF NOT EXISTS tickets(id TEXT PRIMARY KEY,event_id TEXT NOT NULL UNIQUE REFERENCES events(id),status TEXT NOT NULL,repair TEXT,retest TEXT,updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY AUTOINCREMENT,object_id TEXT NOT NULL,action TEXT NOT NULL,detail TEXT NOT NULL,created_at TEXT NOT NULL);
        ''');self.db.commit()

    def audit(self,object_id,action,detail):
        self.db.execute('INSERT INTO audit(object_id,action,detail,created_at) VALUES(?,?,?,?)',(object_id,action,json.dumps(clean(detail),ensure_ascii=False),now()))

    def bind(self,device_id,name,model='电钻'):
        if not device_id.strip() or not name.strip():raise ValueError('设备编号和名称不能为空。')
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO devices VALUES(?,?,?,?)',(device_id,name,model,now()))
        return device_id

    def ingest(self,rows):
        inserted=0
        with self.db:
            for row in rows:
                payload=json.dumps(clean(row),ensure_ascii=False,allow_nan=False)
                cursor=self.db.execute('INSERT OR IGNORE INTO telemetry VALUES(?,?,?,?,?,?)',(row['device_id'],row['run_id'],int(row['sequence']),row['timestamp'],now(),payload))
                if cursor.rowcount==0:continue
                inserted+=1
                if row.get('alarm_confirmed') and row.get('health_state') in ('预警','高风险'):
                    existing=self.db.execute("SELECT id,severity FROM events WHERE device_id=? AND run_id=? AND status!='closed'",(row['device_id'],row['run_id'])).fetchone()
                    if not existing:
                        event_id=uuid.uuid4().hex
                        self.db.execute('INSERT INTO events VALUES(?,?,?,?,?,?,?)',(event_id,row['device_id'],row['run_id'],row['health_state'],'open',now(),payload))
                        self.audit(event_id,'alarm_created',row)
                    elif row['health_state']=='高风险' and existing['severity']!='高风险':
                        self.db.execute('UPDATE events SET severity=?,evidence=? WHERE id=?',('高风险',payload,existing['id']))
                        self.audit(existing['id'],'severity_escalated',row)
        return inserted

    def devices(self):
        devices=[dict(r) for r in self.db.execute('SELECT * FROM devices ORDER BY id')]
        for device in devices:
            row=self.db.execute('SELECT payload,received_at FROM telemetry WHERE device_id=? ORDER BY sampled_at DESC,received_at DESC LIMIT 1',(device['id'],)).fetchone()
            device['latest']=json.loads(row['payload']) if row else None
            if row:device['received_at']=row['received_at']
        return devices

    def history(self,device_id,limit=1000):
        return [json.loads(r['payload']) for r in self.db.execute('SELECT payload FROM (SELECT payload,sampled_at FROM telemetry WHERE device_id=? ORDER BY sampled_at DESC LIMIT ?) ORDER BY sampled_at',(device_id,limit))]

    def events(self):return [dict(r) for r in self.db.execute('SELECT * FROM events ORDER BY created_at DESC')]
    def tickets(self):return [dict(r) for r in self.db.execute('SELECT * FROM tickets ORDER BY updated_at DESC')]

    def acknowledge(self,event_id):
        with self.db:
            cursor=self.db.execute("UPDATE events SET status='acknowledged' WHERE id=? AND status='open'",(event_id,))
            if cursor.rowcount:self.audit(event_id,'acknowledged',{})

    def create_ticket(self,event_id):
        with self.db:
            event=self.db.execute('SELECT * FROM events WHERE id=?',(event_id,)).fetchone()
            if not event or event['status']=='closed':raise ValueError('事件不存在或已关闭。')
            existing=self.db.execute('SELECT id FROM tickets WHERE event_id=?',(event_id,)).fetchone()
            if existing:return existing['id']
            ticket_id=uuid.uuid4().hex
            self.db.execute('INSERT INTO tickets VALUES(?,?,?,?,?,?)',(ticket_id,event_id,'open',None,None,now()))
            self.audit(ticket_id,'ticket_created',{'event_id':event_id})
        return ticket_id

    def repair(self,ticket_id,component,conclusion,operation,evidence):
        if not all(str(x).strip() for x in [component,conclusion,operation,evidence]):raise ValueError('部件、检查结论、维修操作和证据均必填。')
        record={'component':component,'conclusion':conclusion,'operation':operation,'evidence':evidence,'label_status':'pending_review','new_run_id':uuid.uuid4().hex}
        with self.db:
            cursor=self.db.execute("UPDATE tickets SET status='pending_retest',repair=?,updated_at=? WHERE id=? AND status='open'",(json.dumps(record,ensure_ascii=False),now(),ticket_id))
            if not cursor.rowcount:raise ValueError('工单不存在或已进入复测阶段。')
            self.audit(ticket_id,'repair_recorded',record)
        return record

    def retest(self,ticket_id,rows):
        ticket=self.db.execute("SELECT * FROM tickets WHERE id=? AND status='pending_retest'",(ticket_id,)).fetchone()
        if not ticket:raise ValueError('工单必须先完成维修记录。')
        repair=json.loads(ticket['repair']);event=self.db.execute('SELECT * FROM events WHERE id=?',(ticket['event_id'],)).fetchone()
        if len(rows)<5:raise ValueError('至少需要5个连续有效复测结果。')
        recent=rows[-5:]
        valid=all(r['device_id']==event['device_id'] and r['run_id']==repair['new_run_id'] and r.get('validity')=='valid' and r.get('risk_score') is not None and math.isfinite(float(r['risk_score'])) and float(r['risk_score'])<0.25 for r in recent)
        contiguous=all(int(b['sequence'])==int(a['sequence'])+1 for a,b in zip(recent,recent[1:]))
        stable=max(float(r['load']) for r in recent)-min(float(r['load']) for r in recent)<0.15
        passed=bool(valid and contiguous and stable)
        record={'passed':passed,'results':clean(recent),'baseline_update':'frozen_pending_health_confirmation'}
        with self.db:
            self.db.execute('UPDATE tickets SET status=?,retest=?,updated_at=? WHERE id=?',('completed' if passed else 'pending_retest',json.dumps(record,ensure_ascii=False),now(),ticket_id))
            if passed:self.db.execute("UPDATE events SET status='closed' WHERE id=?",(ticket['event_id'],))
            self.audit(ticket_id,'retest_passed' if passed else 'retest_failed',record)
        return passed

    def audit_history(self):return [dict(r) for r in self.db.execute('SELECT * FROM audit ORDER BY id')]
