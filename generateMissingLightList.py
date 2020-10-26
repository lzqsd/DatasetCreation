import glob
import os
import os.path as osp


missingLists = []
srcs = glob.glob('main*')
icnt = 0
for src in srcs:
    scenes = glob.glob(osp.join(src, 'scene*') )
    scenes = sorted(scenes )
    cnt = 0
    for scene in scenes:
        cnt += 1
        imNames = glob.glob(osp.join(scene, 'im_*.hdr') )
        envNames = glob.glob(osp.join(scene, 'imenv_*.hdr') )
        if len(envNames ) < len(imNames ):
            print(cnt, scene )
            icnt += len(imNames ) - len(envNames )

print(icnt )
