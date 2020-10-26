import glob
import os 
import cv2 
import os.path as osp 
import numpy as np 

roots = glob.glob('main*_xml*')
for root in roots:
    scenes = glob.glob(osp.join(root, 'scene*') )
    for scene in scenes:
        print(scene )
        cmd = 'rm %s' % osp.join(scene, 'im*_*_0.h5')
        os.system(cmd ) 

