import os
import json

l1 = os.listdir('/home/sunjiamu/partnet-mobility/data/partnetdata/partnet-mobility-dataset/')
l2 =  os.listdir('/home/yangjie/202/yangjie/structurenet/data/partnetdata/partnetdata_new/')

dic = {'StorageFurniture':0,'Table':0}

idx = 0
for mov in l1:
    for mm in l2:
        if(mm.find(mov) != -1): 
            with open('/home/sunjiamu/partnet-mobility/data/partnetdata/partnet-mobility-dataset/' +mov+'/meta.json') as f:
                jsond = json.load(f)
            if(jsond["model_cat"] == "StorageFurniture"): dic["StorageFurniture"] += 1
            if(jsond["model_cat"] == "Table"): dic["Table"] += 1
            idx += 1
            break



print(idx)
print(len(l1))
print(dic)