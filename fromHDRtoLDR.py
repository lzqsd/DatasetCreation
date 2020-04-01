import cv2
import glob
import os.path as osp
import glob
import numpy as np

cats = glob.glob('*_xml')
sceneIds = glob.glob(osp.join(cats[0], 'scene*' ) )
for sceneId in sceneIds:
    imNum = None

    # Get the image names
    rgbeNames = []
    for cat in cats:
        rgbeNames = rgbeNames + \
                glob.glob(osp.join(cat, sceneId.split('/')[-1], '*.rgbe') )

    if len(rgbeNames ) == 0:
        continue

    imMean = 0
    for rgbeName in rgbeNames:
        im = cv2.imread(rgbeName, -1 )
        imMean +=  np.mean(im )

    imMean /= len(rgbeNames )

    for cat in cats:
        with open(osp.join(cat, sceneId.split('/')[-1], 'mean.txt'), 'w') as fOut:
            fOut.write('%.3f\n' % imMean )

    for rgbeName in rgbeNames:
        im = cv2.imread(rgbeName, -1 )
        im = im / imMean * 0.5
        im = np.clip(im, 0, 1)
        im = im ** (1.0/2.2 )
        im = (255 * im).astype(np.uint8 )

        ldrName = rgbeName.replace('.rgbe', '.png')
        cv2.imwrite(ldrName, im )
