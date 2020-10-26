import glob
import os.path as osp 
import os 

srcRoots = glob.glob('missingLight/main*')
for srcRoot in srcRoots:
    root = srcRoot.split('/')[-1] 
    scenes = glob.glob(osp.join(srcRoot, 'scene*') )
    for scene in scenes:
        sceneId = scene.split('/')[-1]
        dst = osp.join(root, sceneId )

        envNames = glob.glob(osp.join(scene, 'imenv_*.hdr') )
        for envName in envNames:
            if osp.getsize(envName ) > 100:
                print(envName )
                envId = envName.split('/')[-1] 
                dstName = osp.join(dst, envId) 
                print(dstName )
                assert(not osp.isfile(dstName ) ) 
                os.system('mv %s %s' % (envName, dstName  ) )
