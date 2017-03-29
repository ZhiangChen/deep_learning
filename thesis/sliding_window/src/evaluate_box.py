import numpy as np
import os

from evaluator import *
from orthaffine import OrthAffine as OA
os.chdir('../png')

files = os.listdir('.')
theta = 30.0/180*3.14
oa = OA(theta)
ev = evaluator()

names = list()
scores = list()
for f in files:
    if f.split('.')[-1]=='pcd':
        oa.readpcd(f)
        image = oa.project_small().reshape(-1,40,40,1).astype(np.float32)
        images = np.concatenate((image,small_data),axis=0)
        c,s,a = ev.evaluate(images)
        name = name2string[value2name[int(c[0,0])]]
        names.append(name)
        scores.append(str(s[0,0]))

with open('file','w') as f:
	for i,name in enumerate(names):
		f.write(name+' '+scores[i]+'\n')

