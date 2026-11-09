from app import app, jobs
import json


def test_trial_selector_is_strict_browser_json():
    # Real training annotations contain a missing-text NaN. Python's default
    # decoder tolerates it, but browser Response.json() rejects the whole list.
    def reject_constant(value):
        raise ValueError('Non-JSON numeric constant: '+value)
    with app.test_client() as c:
        response=c.get('/api/criteria')
        rows=json.loads(response.get_data(as_text=True),parse_constant=reject_constant)
        assert response.status_code==200 and rows
        assert all(isinstance(r['text'],str) and 1<=len(r['text'].strip())<=3000
                   and r['type'] in ('inclusion','exclusion') for criteria in rows.values() for r in criteria)

def test_invalid_requests():
    with app.test_client() as c:
        assert c.get('/').status_code == 200
        assert c.post('/api/screen',json={'patient':'','criteria':[]}).status_code == 400
        assert c.post('/api/screen',data='text').status_code == 415
        assert c.post('/api/screen',json={},headers={'Origin':'https://other.example'}).status_code == 403
        assert c.get('/api/export/unknown').status_code == 404
        assert c.post('/api/screen',data='x'*140000,content_type='application/json').status_code == 413

def test_export_escapes_input():
    jobs['test']={'status':'done','result':{'criteria':[],'notice':'test',
                  'patient_record':'<script>alert(1)</script>','overall_preliminary_outcome':'review'}}
    try:
        with app.test_client() as c:
            r=c.get('/api/export/test')
            assert r.status_code==200
            assert b'&lt;script&gt;' in r.data and b'<script>' not in r.data
    finally:
        del jobs['test']
