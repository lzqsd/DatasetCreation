import cv2
import glob
import os.path as osp
import glob
import numpy as np

scenes = glob.glob('Images/scene*')
for scene in scenes:
    hdrs = glob.glob(osp.join(scene, '*.rgbe') )
    for hdr in hdrs:
        im = cv2.imread(hdr, -1)
        im = im / np.mean(im ) * 0.5
        im = np.clip(im ** (1.0/2.2), 0, 1)
        im = (255 * im).astype(np.uint8 )
        ldr = hdr.replace('.rgbe', '.png')
        cv2.imwrite(ldr, im)
