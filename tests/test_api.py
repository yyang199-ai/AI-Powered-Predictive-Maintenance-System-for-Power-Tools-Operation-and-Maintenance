import os
import tempfile
import unittest
from fastapi.testclient import TestClient
from predictive.api import app

class APITests(unittest.TestCase):
    def test_local_device_binding_and_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            original=os.environ.get('MAINTENANCE_DB')
            os.environ['MAINTENANCE_DB']=folder+'/api.sqlite3'
            try:
                with TestClient(app) as client:
                    self.assertEqual(client.get('/health').json()['status'],'ok')
                    self.assertEqual(client.post('/devices',json={'device_id':'D1','name':'电钻'}).status_code,200)
                    self.assertEqual(len(client.get('/devices').json()),1)
                    self.assertEqual(client.get('/devices/D1/history').json(),[])
                    self.assertEqual(client.get('/events/missing').status_code,404)
                    self.assertEqual(client.post('/devices',json={'device_id':'','name':'x'}).status_code,422)
                    self.assertTrue(client.get('/models').json())
                    self.assertEqual(client.get('/openapi.json').status_code,200)
                    from predictive.data import simulate_fleet
                    import json
                    data=simulate_fleet(1,40).assign(device_id='D1')
                    rows=json.loads(data.to_json(orient='records'))
                    response=client.post('/telemetry',json={'rows':rows})
                    self.assertEqual(response.status_code,200,response.text)
                    self.assertEqual(response.json()['inserted'],40)
                    response=client.post('/telemetry',json={'rows':rows})
                    self.assertEqual(response.json()['inserted'],0)
                    self.assertEqual(len(client.get('/devices/D1/history').json()),40)
            finally:
                if original is None:os.environ.pop('MAINTENANCE_DB',None)
                else:os.environ['MAINTENANCE_DB']=original
