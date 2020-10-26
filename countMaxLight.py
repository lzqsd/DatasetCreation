import glob 
import os
import os.path as osp 
import cv2
import numpy as np

scenes = glob.glob('main_xml/scene*')
maxNum = 0 
for scene in scenes:
    lightDirs = glob.glob(osp.join(scene, 'light_*') )
    for lightDir in lightDirs:
        maskNames = glob.glob(osp.join(lightDir, 'mask*.png') )
        num = 0
        for maskName in maskNames:
            im = cv2.imread(maskName )
            im = im.astype(np.float32 )
            if np.sum(im) == 0:
                num += 1
        if num > maxNum:
            maxNum = num
            print(lightDir, num )


print(maxNum )
