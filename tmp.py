import os
import json

with open('./mesh_matching/replacefinalfinalnew.json') as f:
    replacement = json.load(f)

moveable_list = os.listdir('B:\\partnet-mobility-v0_2\\dataset')

l = []

for v in replacement.values():
    for item in v:
        l.append(item['name'].split('/')[0])

l = list(set(l))
#print(l)
#print(moveable_list)
print(len(list(set(l).intersection(set(moveable_list)))))