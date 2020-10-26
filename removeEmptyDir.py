import glob
import os 
import os.path as osp 

roots = glob.glob('main*')
for root in roots:
    scenes = glob.glob(osp.join(root, 'scene*') )
    cnt = 0
    for scene in scenes:
        hdrs = glob.glob(osp.join(scene, '*.hdr') )
        if len(hdrs ) == 0:
            cnt += 1 
            os.system('rm -r %s' % scene )
    print(root, cnt )
