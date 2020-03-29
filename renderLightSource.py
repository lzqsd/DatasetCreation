import os
import os.path as osp
import glob
import argparse
import time
import struct
import numpy as np
import cv2
import open3d as o3d
from sklearn.cluster import KMeans
from torch.autograd import Variable
import torch
import numpy.matlib
from scipy import ndimage
import h5py
import OpenEXR
import Imath
import array

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train')
parser.add_argument('--out', default='./xml', help='path to the image root')
parser.add_argument('--dst', default='./Images', help='path to save the model')
parser.add_argument('--program', \
        default='/home/zhl/CVPR20/LightEditing/OptixRendererDirectLightPara/src/bin/optixRenderer' )
parser.add_argument('--gpuIds', nargs='+', default = None, help = 'Gpu Ids for rendering' )
parser.add_argument('--rs', default = 0, type=int, help='the starting point' )
parser.add_argument('--re', default = 1, type=int, help='the end point' )
parser.add_argument('--cs', default = 0, type=int, help='the start point of camera view' )
parser.add_argument('--ce', default = -1, type=int, help='the end point of camera view' )
parser.add_argument('--panHeight', default=256, type=int, help='the height of the panorama')
parser.add_argument('--panWidth', default=512, type=int, help='the width of the panorama')
parser.add_argument('--fov', default=57.0, type=float, help='the field of vview')
parser.add_argument('--isVisualize', action='store_true', help = 'visualize the rendered first bounce intensity')
opt = parser.parse_args()
print(opt)

program = opt.program
panHeight = opt.panHeight
panWidth = opt.panWidth
isVisualize = opt.isVisualize

t1 = time.time()

scenes = glob.glob(osp.join(opt.out, 'scene*') )
scenes = [x for x in scenes if osp.isdir(x) ]
scenes = sorted(scenes )
for n in range(opt.rs, min(opt.re, len(scenes ) ) ):
    scene = scenes[n]
    sceneId = scene.split('/')[-1]

    print('%d/%d: %s' % (n, len(scenes), sceneId ) )

    xmlFile = osp.join(scene, 'main.xml'  )
    if not osp.isfile(xmlFile ):
        continue

    dstDir = osp.join(opt.dst, sceneId )
    if not osp.isdir(dstDir ):
        continue

    dstDir = osp.join(os.getcwd(), dstDir )

    originArr, lookAtArr, upArr = [], [], []
    camFile = osp.join( scene, 'cam.txt' )
    with open(camFile, 'r') as camIn:
        camNum = int(camIn.readline().strip() )
        for camId in range(camNum ):
            originStr = camIn.readline().strip().split(' ')
            lookAtStr = camIn.readline().strip().split(' ')
            upStr = camIn.readline().strip().split(' ')
            origin = np.asarray([float(x) for x in originStr ], dtype=np.float32 )
            lookAt = np.asarray([float(x) for x in lookAtStr ], dtype=np.float32 )
            up = np.asarray([float(x) for x in upStr ], dtype=np.float32 )

            originArr.append(origin )
            lookAtArr.append(lookAt )
            upArr.append(up )

    for m in range(0, camNum ):
        camId = m + 1
        # Locate the light sources
        cameraLoc = originArr[m]

        # Compute the camera locations
        lookAt = lookAtArr[m]
        up = upArr[m]
        zAxis = (cameraLoc - lookAt )
        zAxis = zAxis / np.sqrt(np.sum(zAxis * zAxis ) )
        yAxis = up / np.sqrt(np.sum(up * up ) )
        xAxis = np.cross(yAxis, zAxis )
        xAxis = xAxis / np.sqrt(np.sum(xAxis * xAxis ) )
        yAxis = np.cross(zAxis, xAxis )
        rot = np.concatenate([xAxis.reshape(3, 1), \
                yAxis.reshape(3, 1), zAxis.reshape(3, 1) ], axis=1 )

        # Load the depth and the mask
        maskName = osp.join(dstDir, 'immask_%d.png' % camId )
        mask = cv2.imread(maskName )[:, :, 0]
        mask = cv2.resize(mask, (160, 120), interpolation = cv2.INTER_AREA )
        mask = (mask > 0.45)

        depthName = osp.join(dstDir, 'imdepth_%d.dat' % camId )
        with open(depthName, 'rb') as fIn:
            hBuffer = fIn.read(4)
            height = struct.unpack('i', hBuffer)[0]
            wBuffer = fIn.read(4)
            width = struct.unpack('i', wBuffer)[0]
            dBuffer = fIn.read(4 * width * height )
            depth = np.asarray(struct.unpack('f' * height * width, dBuffer), dtype=np.float32 )
            depth = depth.reshape([height, width] )
        depth = cv2.resize(depth, (160, 120), interpolation = cv2.INTER_AREA )

        # Compute the sampled locations
        xRange = np.tan(opt.fov / 180.0 * np.pi / 2.0 )
        yRange = float(120 ) / float(160 ) * xRange
        x, y = np.meshgrid(np.linspace(-xRange, xRange, 160),
                np.linspace(-yRange, yRange, 120 ) )
        y = np.flip(y, axis=0)
        z = -np.ones( (120, 160 ), dtype=np.float32 )
        pCoord = np.stack([x, y, z] ).astype(np.float32 )

        pCoord = pCoord * depth[np.newaxis, :]
        pCoord = pCoord.reshape(3, 120 * 160 )
        pCoord = np.matmul(rot, pCoord ) + cameraLoc.reshape(3, 1)

        mask = mask.flatten()
        pCoord = pCoord[:, mask == 1]
        selectedInd = np.random.permutation(pCoord.shape[1] )
        panLocs =  pCoord[:, selectedInd[0:min(80, pCoord.shape[1]) ] ]

        panLocs = panLocs * 0.9 + cameraLoc.reshape(3, 1) * 0.1

        # Render the camera light points
        with open(osp.join(scene, 'cam%dLight.txt' % camId  ), 'w') as camOut:
            camOut.write('%d\n' % panLocs.shape[1] )
            for n in range(0, panLocs.shape[1] ):
                origin = panLocs[:, n]
                up = upArr[m]
                zAxis = cameraLoc - origin
                zAxis = zAxis / np.sqrt(np.sum(zAxis * zAxis ) )
                lookAt = origin + zAxis * 1
                up = up - (up * zAxis) * zAxis
                up = up / np.sqrt(np.sum(up * up ) )
                camOut.write('%.3f %.3f %.3f\n' % (origin[0], origin[1], origin[2] ) )
                camOut.write('%.3f %.3f %.3f\n' % (lookAt[0], lookAt[1], lookAt[2] ) )
                camOut.write('%.3f %.3f %.3f\n' % (up[0], up[1], up[2] ) )
        outFile = osp.join(dstDir, 'imPan.rgbe' )
        os.system('{0} -f {1} -o {2} -m {3} -c {4} --forceOutput'.format(
            program, xmlFile, outFile, 7, 'cam%dLight.txt' %  camId ) )

        os.system('rm %s' % (osp.join(scene, 'cam%dLight.txt' % camId ) ) )
        panFiles = glob.glob(osp.join(dstDir, 'imPan*.dat' ) )

        pIntensities = []
        pDirections = []
        pPositions = []
        for panFile in panFiles:
            with open(panFile, 'rb') as fIn:
                panBuffer = fIn.read()
            os.system('rm %s' % panFile )
            pan = struct.unpack(str(panHeight * panWidth * 9) + 'f', panBuffer )
            pan = np.array(pan, dtype=np.float32 ).reshape([panHeight, panWidth, 9] )
            position = pan[:, :, 0:3 ]
            pIntensity = pan[:, :, 3:6 ]
            pDirection = pan[:, :, 6:9 ]

            image = np.clip(pIntensity, 0, 1)
            image = (255 * (image ** (1.0/2.2) ) ).astype(np.uint8 )

            mask = (np.sum(pIntensity, axis=2 ) > 0)
            mask = ndimage.binary_erosion(mask, border_value=1, \
                    structure = np.ones((4, 4) ) )
            mask = mask.flatten()

            position = position.reshape(-1, 3)
            pIntensity = pIntensity.reshape(-1, 3)
            pDirection = pDirection.reshape(-1, 3)
            position = position[mask==1, :]
            pIntensity = pIntensity[mask == 1, :]
            pDirection = pDirection[mask == 1, :]

            pPositions.append(position )
            pIntensities.append(pIntensity )
            pDirections.append(pDirection )

        pPositions = np.concatenate(pPositions, axis=0 )[0::5, :]
        pIntensities = np.concatenate(pIntensities, axis=0 )[0::5, :]
        pDirections = np.concatenate(pDirections, axis=0 )[0::5, :]

        pPositions = pPositions - cameraLoc.reshape(1, 3)
        pPositions = np.matmul(rot.transpose(1, 0), pPositions.transpose(1, 0) )
        pPositions = pPositions.transpose(1, 0)

        pDirections = np.matmul(rot.transpose(1, 0), pDirections.transpose(1, 0) )
        pDirections = pDirections.transpose(1, 0)

        if isVisualize:
            pcd = o3d.geometry.PointCloud()
            pcdName = osp.join(opt.dst, sceneId, 'im_%dlight.ply' % camId )
            pcd.points = o3d.utility.Vector3dVector(pPositions  )
            pcd.normals = o3d.utility.Vector3dVector(pDirections )
            pcd.colors = o3d.utility.Vector3dVector(pIntensities )
            o3d.io.write_point_cloud(pcdName, pcd)


t2 = time.time()
print('Hours: %.3f' %  ( (t2 - t1) / 3600.0 ) )
