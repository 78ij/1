import os
import json
from common import *

with open(FUTURE_PATH + '/model_info.json') as f:
    json_data = json.load(f)

cates = []
for entry in json_data:
    cates.append(entry['category'])

print(list(set(cates)))