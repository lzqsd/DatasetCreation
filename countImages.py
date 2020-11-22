import glob 
import os.path as osp 

roots = glob.glob('main_xml*')
hdrNum = 0
for root in roots:
    scenes = glob.glob(osp.join(root, 'scene*') )
    for scene in scenes: 
        if 'scene0422_00' in scene:
            continue
        hdrNum += len(glob.glob(osp.join(scene, 'im_*.hdr') ) )
print(hdrNum )

