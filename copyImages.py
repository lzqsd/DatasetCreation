import glob
import os
import os.path as osp
import cv2  
import random 
import numpy as np


dst = 'sampled_images_test_1' 
os.system('mkdir %s' % dst )  
maxNum =3 

with open('test.txt', 'r') as fIn:
    testScenes = fIn.readlines() 
testScenes = [x.strip() for x in testScenes ]

dirs = glob.glob('main*_xml1') 
cnt = 0 
for scene in testScenes:
    cnt += 1
    hdrNames = glob.glob(osp.join(dirs[0], scene, 'im_*.hdr') )
    hdrNames = [x.split('/')[-1] for x in hdrNames ] 
    random.shuffle(hdrNames )
    hdrNames = hdrNames[0: min(maxNum, len(hdrNames ) ) ]

    imgs = [] 
    for d in dirs:
        for hdrName in hdrNames:
            hdrName = osp.join(d, scene, hdrName )
            im = cv2.imread(hdrName, -1 )
            imgs.append(im ) 

    imMean = 0 
    for im in imgs:
        imMean += np.mean(im )
    imMean = imMean / len(hdrNames )
    

    for m in range(0, len(dirs) ):
        for n in range(0, len(hdrNames ) ):
            d = dirs[m]
            hdrName = hdrNames[n]
            im = imgs[m*len(hdrNames) + n ]

            im = im / imMean * 0.5 
            im = (np.clip(im, 0, 1 ) ** (1.0/2.2) * 255).astype(np.uint8 )
            imName = scene + '-' + hdrName.split('.')[0] + '-' + d + '.png'
            imName = osp.join(dst, imName )
            cv2.imwrite(imName, im )
            
    
