import glob
import os
import os.path as osp
import h5py
import numpy as np
import cv2

root = 'Images'
imHeight = 120
imWidth = 160
envWidth = 32
envHeight = 16

scenes = glob.glob(osp.join(root, 'scene*') )
for scene in scenes:
    h5fs = glob.glob(osp.join(scene, '*.h5') )
    for h5f in h5fs:
        print(h5f )
        hf = h5py.File(h5f, 'r')
        envs = np.asarray(hf.get('env'), dtype=np.float32 )

        envNewName = h5f.replace('.h5', '.hdr' )
        envs = envs.transpose([0, 2, 1, 3, 4] )
        envs = envs.reshape(imHeight * envHeight, imWidth * envWidth, 3 )
        cv2.imwrite(envNewName, envs )

        os.system('rm %s' % h5f )
