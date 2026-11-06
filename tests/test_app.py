from app import app, jobs

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
