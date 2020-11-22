import os
import os.path as osp
import glob


os.system('mkdir samples')
dirs = glob.glob('main*')
for d in dirs:
    os.system('mkdir -p %s' % osp.join('samples', d ) )
    os.system('cp -r %s %s' % (osp.join(d, 'scene0006_01'),
        osp.join('samples', d) ) )
    os.system('rm %s' % (osp.join('samples', d, 'scene0006_01/imenv_*.hdr') ) )
