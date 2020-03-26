import glob
import os
import os.path as osp
import argparse
import h5py
import struct
import numpy as np
import OpenEXR
import Imath
import array
import cv2

parser = argparse.ArgumentParser()
# Directories
parser.add_argument('--out', default="./xml/", help="outdir of xml file")
# Start and end point
parser.add_argument('--rs', default=0, type=int, help='the width of the image' )
parser.add_argument('--re', default=1600, type=int, help='the height of the image' )
# Control
parser.add_argument('--forceOutput', action='store_true', help='whether to overwrite previous results')
parser.add_argument('--medianFilter', action='store_true', help='whether to use median filter')
# output dir
parser.add_argument('--outputDir', default='ImagesDiffLight', help='the output dir')
# Program
parser.add_argument('--program', default='~/OptixRendererLight/src/bin/optixRenderer', help='the location of render' )
opt = parser.parse_args()

imHeight = 120
imWidth = 160
envWidth = 32
envHeight = 16
pixelNum = imHeight * imWidth * envHeight * envWidth * 3

scenes = glob.glob(osp.join(opt.out, 'scene*') )
scenes = [x for x in scenes if osp.isdir(x) ]
scenes = sorted(scenes )
for n in range(opt.rs, min(opt.re, len(scenes ) ) ):
    scene = scenes[n]
    sceneId = scene.split('/')[-1]

    print('%d/%d: %s' % (n, len(scenes), sceneId ) )

    outDir = osp.join(os.getcwd(), opt.outputDir, sceneId )
    os.system('mkdir -p %s' % outDir )

    xmlFile = osp.join(scene, 'mainDiffLight.xml' )
    camFile = osp.join(scene, 'cam.txt' )
    if not osp.isfile(xmlFile ) or not osp.isfile(camFile ):
        continue

    cmd = '%s -f %s -c %s -o %s -m %d' % (opt.program, xmlFile, 'cam.txt', osp.join(outDir, 'im.rgbe'), 7)

    if opt.forceOutput:
        cmd += ' --forceOutput'

    os.system(cmd )

    envNames = glob.glob(osp.join(outDir, 'imenv_*.dat') )
    for envName in envNames:
        with open(envName, 'rb') as envIn:
            byteArr = envIn.read()
        byteArr = struct.unpack('%df' % pixelNum, byteArr )
        byteArr = np.array(byteArr, dtype=np.float32 )
        byteArr = byteArr.reshape(imHeight, imWidth, envHeight, envWidth, 3 )

        envNewName = envName.replace('.dat', 'weird.hdr')
        envs = byteArr.transpose([0, 2, 1, 3, 4] )
        envs = envs.reshape(imHeight * envHeight, imWidth * envWidth, 3 )
        cv2.imwrite(envNewName, envs )

        envNewName = envName.replace('.dat', 'weird.png')
        envs = byteArr.transpose([0, 2, 1, 3, 4] )
        envs = envs / np.mean(envs ) * 0.5
        envs = (np.clip(envs, 0, 1) * 255 ).astype(np.uint8 )
        envs = envs.reshape(imHeight * envHeight, imWidth * envWidth, 3 )
        cv2.imwrite(envNewName, envs )

        #os.system('rm %s' % envName )

    break
