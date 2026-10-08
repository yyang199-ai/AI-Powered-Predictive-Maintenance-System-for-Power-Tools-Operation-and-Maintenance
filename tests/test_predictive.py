import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from predictive.data import simulate_fleet, grouped_split, FEATURES, QUALITY
from predictive.preprocessing import ConditionBaseline
from predictive.pipeline import build_sequences, predict_trajectory
from predictive.alarms import AlarmState, annotate_alarms
from predictive.store import Store
from predictive.signals import waveform_features, quality_score

torch.set_num_threads(2)

class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=simulate_fleet(10,100)
        cls.splits=grouped_split(cls.data)
        cls.baseline=ConditionBaseline().fit(cls.splits['train'])

    def test_device_split_and_frozen_baseline(self):
        sets=[set(d.device_id) for d in self.splits.values()]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        before=self.baseline.to_dict()
        self.baseline.transform(self.splits['test'])
        self.assertEqual(before,self.baseline.to_dict())

    def test_missing_and_unknown_conditions(self):
        row=self.data.iloc[:1].copy();row[FEATURES]=np.nan
        f,q,c,unknown=self.baseline.transform(row)
        self.assertTrue((q==0).all());self.assertTrue(np.isfinite(f).all())
        row['mode']='impact'
        self.assertTrue(self.baseline.transform(row)[3].all())

    def test_sequences_never_cross_missing_or_gaps(self):
        data=self.data[self.data.device_id=='DRILL-001'].copy()
        data.loc[data.sequence==25,QUALITY]=0
        inputs,targets,indices=build_sequences(data,self.baseline)
        for index in indices:
            self.assertFalse(index-15<=25<=index)
        data['run_id']=[f'run-{i}' for i in range(len(data))]
        with self.assertRaises(ValueError):build_sequences(data,self.baseline)

    def test_causal_waveform_and_quality(self):
        rate=1024;x=np.sin(2*np.pi*64*np.arange(2048)/rate)
        features,state=waveform_features(x,rate)
        self.assertAlmostEqual(features['spectral_peak_hz'],64,delta=1)
        self.assertEqual(quality_score(x,np.arange(len(x))/rate,rate),1)
        self.assertEqual(quality_score(np.ones(20),np.arange(20)/rate,rate),0)
        times=np.arange(len(x))/rate;times[30:]+=0.1
        self.assertEqual(quality_score(x,times,rate),0)

class AlarmTests(unittest.TestCase):
    def test_confirmation_gaps_and_invalid(self):
        state=AlarmState()
        def row(i,risk=0.9,validity='valid',run='a'):
            return dict(sequence=i,run_id=run,load=0.5,risk_score=risk,validity=validity)
        self.assertFalse(state.update(row(0))['confirmed'])
        self.assertFalse(state.update(row(1))['confirmed'])
        self.assertTrue(state.update(row(2))['confirmed'])
        self.assertEqual(state.update(row(3,validity='insufficient_data'))['state'],'数据不足')
        self.assertFalse(state.update(row(20))['confirmed'])
        self.assertFalse(state.update(row(21,run='b'))['confirmed'])

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'db.sqlite3'
        self.store=Store(self.path);self.store.bind('a','电钻A')
        self.row=dict(device_id='a',run_id='old',sequence=1,timestamp='2026-01-01T00:00:00+00:00',health_state='预警',alarm_confirmed=True,risk_score=0.6,load=0.5,validity='valid')

    def tearDown(self):self.store.db.close();self.temp.cleanup()

    def test_dedup_persistence_and_repair_closure(self):
        self.assertEqual(self.store.ingest([self.row,self.row]),1)
        event=self.store.events()[0];self.store.acknowledge(event['id'])
        self.assertEqual(self.store.events()[0]['status'],'acknowledged')
        ticket=self.store.create_ticket(event['id']);self.assertEqual(ticket,self.store.create_ticket(event['id']))
        with self.assertRaises(ValueError):self.store.retest(ticket,[])
        repair=self.store.repair(ticket,'轴承','模拟确认磨损','模拟更换','仿真证据')
        old=[{**self.row,'sequence':i,'risk_score':0.1} for i in range(5)]
        self.assertFalse(self.store.retest(ticket,old))
        new=[{**r,'run_id':repair['new_run_id']} for r in old]
        self.assertTrue(self.store.retest(ticket,new))
        self.assertEqual(self.store.events()[0]['status'],'closed')
        reopened=Store(self.path);self.assertEqual(reopened.tickets()[0]['status'],'completed');reopened.db.close()

    def test_event_escalation_does_not_reopen_ack(self):
        self.store.ingest([self.row]);event=self.store.events()[0];self.store.acknowledge(event['id'])
        self.store.ingest([{**self.row,'sequence':2,'health_state':'高风险'}])
        self.assertEqual(len(self.store.events()),1)
        self.assertEqual(self.store.events()[0]['severity'],'高风险')
        self.assertEqual(self.store.events()[0]['status'],'acknowledged')

class ModelTests(unittest.TestCase):
    def test_rejects_invalid_identity_and_time(self):
        data=simulate_fleet(1,40,seed=100)
        broken=data.copy();broken.loc[0,'sequence']=0.5
        with self.assertRaises(ValueError):predict_trajectory(broken)
        duplicate=data.iloc[[0,0]].copy()
        with self.assertRaises(ValueError):predict_trajectory(duplicate)

    def test_bundle_integrity_check(self):
        import shutil
        from predictive.pipeline import load_bundle
        with tempfile.TemporaryDirectory() as folder:
            shutil.copytree('artifacts/model',Path(folder)/'model')
            weights=Path(folder)/'model/weights.pt'
            weights.write_bytes(weights.read_bytes()+b'corrupt')
            with self.assertRaises(ValueError):load_bundle(Path(folder)/'model')

    def test_bundle_forecast_and_all_missing_abstention(self):
        data=simulate_fleet(1,100,seed=66)
        result=predict_trajectory(data)
        self.assertTrue(result.health_index.notna().any())
        valid=result.dropna(subset=['rul_lower_hours'])
        self.assertTrue((valid.rul_lower_hours<=valid.rul_median_hours).all())
        self.assertTrue((valid.rul_median_hours<=valid.rul_upper_hours).all())
        data[FEATURES]=np.nan;data[QUALITY]=0
        result=predict_trajectory(data)
        self.assertTrue((result.validity=='insufficient_data').all())
        self.assertTrue(result.health_index.isna().all())

if __name__=='__main__':unittest.main()
