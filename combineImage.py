import glob
import os
import os.path as osp
import cv2


srcs = glob.glob('main*_xml*')
for src in srcs:
    scenes = glob.glob(osp.join(src, 'scene*') )
    for scene in scenes:
        print(scene )
        imsNames = glob.glob(osp.join(scene, 'ims_*.rgbe') )

        if len(imsNames ) != 0:
            for imsName in imsNames:
                imName = imsName.replace('ims_', 'im_')
                im = cv2.imread(imName, -1)
                ims = cv2.imread(imsName, -1)
                imn = 1.0/3.0 * im + 2.0/3.0 * ims

                imnName = imName.replace('.rgbe', '.hdr')
                cv2.imwrite(imnName, imn )

                os.system('rm %s' % imName )
                os.system('rm %s' % imsName )
        else:
            imNames = glob.glob(osp.join(scene, 'im_*.rgbe') )
            for imName in imNames:
                imNewName = imName.replace('.rgbe', '.hdr')
                os.system('mv %s %s' % (imName, imNewName ) )
