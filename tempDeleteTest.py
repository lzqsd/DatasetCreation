import glob
import os 
import os.path as osp 
import numpy as np 


with open('test.txt', 'r') as fIn:
    sceneList = fIn.readlines()
sceneList = [x.strip() for x in sceneList ]

roots = glob.glob('main*_xml*')
for root in roots:
    if osp.isdir(root ):
        cnt = 0
        for scene in sceneList:
            cnt += 1 
            cmd = 'rm -r %s' % osp.join(root, scene )  
            if osp.isdir(osp.join(root, scene ) ):
                print(cnt, cmd )
                os.system(cmd )
                

