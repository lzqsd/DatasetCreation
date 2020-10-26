import glob
import os
import os.path as osp


missingLists = []
srcs = glob.glob('main*')
srcs = [x for x in srcs if osp.isdir(x) ]
for src in srcs:
    scenes = glob.glob(osp.join(src, 'scene*') )
    scenes = sorted(scenes )
    with open(src + '.txt', 'w') as fOut:
        for scene in scenes: 
            envNames = glob.glob(osp.join(scene, 'imenv_*.hdr') )  
            envDirectNames = glob.glob(osp.join(scene, 'imenvDirect_*.hdr') ) 
            if len(envNames) != len(envDirectNames ):
                fOut.write('%s\n' % scene )
                print(scene )

