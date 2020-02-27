import pandas as pd
import json
from datetime import datetime

#data = pd.read_json('Data/views.json')


with open('Data/views.json') as json_file:
    data = json_file.readlines()
    data = list(map(json.loads, data))
    
data = pd.DataFrame(data)

print(data)
print(data['userIP'])
print(data['userAgent'])

data = data[:50000]

for i in range(len(data['timestamp'])):
    long = int(data['timestamp'][i]['$date']['$numberLong'])
    dt = datetime.fromtimestamp(long/1000)
    data['timestamp'][i] = dt
    
print(data['timestamp'])
print(data)