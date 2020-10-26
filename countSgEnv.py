import glob
import os.path as osp 

roots = glob.glob('main*_xml*')
cnt = 0
for root in roots:
    scenes = glob.glob(osp.join(root, 'scene*') )
    for scene in scenes:
        sgs = glob.glob(osp.join(scene, 'imsgEnv*.h5') )
        cnt = cnt + len(sgs )
print(cnt )

