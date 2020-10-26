import os.path as osp 
import glob 
import random

scenes = glob.glob(osp.join('main_xml', 'scene*') )
scenes = [x.split('/')[-1] for x in scenes ]
random.shuffle(scenes )

testScenes = scenes[0:110 ]
trainScenes = scenes[110:]

with open('test.txt', 'w') as fOut:
    for s in testScenes:
        fOut.write('%s\n' % s )

with open('train.txt', 'w') as fOut:
    for s in trainScenes:
        fOut.write('%s\n' % s )
