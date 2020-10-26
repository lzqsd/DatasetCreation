import glob 
import os.path as osp 

roots = glob.glob('main*_xml*')
hdrNum = 0
for root in roots:
    scenes = glob.glob(osp.join(root, 'scene*') )
    for scene in scenes:
        hdrNum += len(glob.glob(osp.join(scene, 'im_*.hdr') ) )
print(hdrNum )
