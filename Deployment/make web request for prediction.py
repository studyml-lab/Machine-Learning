import json,requests

data = '[1.0,1.0,0.5,1.0,1.0,2.1,1.2,1.0]'

res = requests.post('http://127.0.0.1:9000/api',data)

res.json()
