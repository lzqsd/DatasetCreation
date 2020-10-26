import os
import os.path as osp
import glob


os.system('mkdir samples')
dirs = glob.glob('main*')
for d in dirs:
    os.system('mkdir %s' % osp.join('samples', d) )
    os.system('cp -r %s %s' % (osp.join(d, 'scene0001_01'), osp.join('samples', d ) ) )
