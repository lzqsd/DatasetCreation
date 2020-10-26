import glob
import os
import os.path as osp


dstRoot = 'missingLight'

srcs = glob.glob('main*')
for src in srcs:
    dst = osp.join(dstRoot, src )
    xmlRoot = osp.join('../scenes', src.split('_')[1] )
    scenes = glob.glob(osp.join(xmlRoot, 'scene*') )
    scenes = sorted(scenes )
    scenes = [x for x in scenes if osp.isdir(x) ]

    missingLists = []
    for n in range(0, 8 ):
        missingList = []
        tempList = []
        for m in range(n*200, min(len(scenes ), (n+1) * 200 ) ):
            scene = scenes[m]
            sceneId = scene.split('/')[-1]

            srcDir = osp.join(src, sceneId )
            envs = glob.glob(osp.join(srcDir, 'imenv*.hdr') )
            camFile = osp.join(scene, 'cam.txt')
            if not osp.isfile(camFile ):
                continue
            with open(camFile, 'r') as camIn:
                camNum = int(camIn.readline().strip() )

            if len(envs ) != 0:
                if len(tempList ) != 0:
                    missingList = missingList + tempList
                    tempList = []

            if len(envs ) != camNum:
                envList = []
                for n in range(1, camNum + 1):
                    envName = osp.join(srcDir, 'imenv_%d.hdr' % n )
                    if not osp.isfile(envName ):
                        envList.append(envName )

                if len(envList ) > 0:
                    tempList.append(envList )

        missingLists += missingList

    dst = osp.join(dstRoot, src )
    os.system('mkdir -p %s' % dst )
    for missingList in missingLists:
        sceneId = missingList[0].split('/')[1]

        dstDir = osp.join(dst, sceneId )
        os.system('mkdir -p %s' % dstDir )

        camFile = osp.join(xmlRoot, sceneId, 'cam.txt')
        assert(osp.isfile(camFile ) )
        with open(camFile, 'r') as camIn:
            camNum = int(camIn.readline().strip() )

        for n in range(1, camNum + 1):
            print(osp.join(dstDir, 'imenv_%d.hdr' % n ) )
            with open(osp.join(dstDir, 'imenv_%d.hdr' % n ), 'w') as fOut:
                fOut.write('hehe')

        for name in missingList:
            name  = osp.join(dstRoot, name )
            os.system('rm %s' % name )

