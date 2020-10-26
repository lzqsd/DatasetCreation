import glob
import os
import os.path as osp 

os.system('mkdir data')
dirs = glob.glob('main*_xml*')
for d in dirs:
    scenes = glob.glob(osp.join(d, 'scene*') )
    for scene in scenes:
        dst = osp.join('data',scene ) 
        os.system('mkdir -p %s' % dst )

        os.system('cp %s %s' % (osp.join(scene, '*.png'), dst) )
        os.system('cp %s %s' % (osp.join(scene, '*.dat'), dst) )
        os.system('cp %s %s' % (osp.join(scene, 'im_*.hdr'), dst) ) 

