import numpy as np
import os
import shutil
import glob
import JSONHelper
import quaternion
import argparse
import os.path as osp
import pickle
import align_utils as utils
import xml.etree.ElementTree as et
from xml.dom import minidom
import cv2
import struct
import scipy.ndimage as ndimage


def computeCameraEx(rotMat, trans, scale_scene = None, rotMat_scene = None, trans_scene = None ):
    view = np.array([0, 0, 1], dtype = np.float32 )
    up = np.array([0, -1, 0], dtype = np.float32 )

    rotMat = rotMat.reshape(3, 3)
    view = view.reshape(1, 3)
    up = up.reshape(1, 3)
    trans = trans.squeeze()

    origin = trans
    lookat = np.sum(rotMat * view, axis=1) + origin
    up = np.sum(rotMat * up, axis=1 )

    if not (scale_scene is None or rotMat_scene is None):
        origin = np.sum((origin * scale_scene)[np.newaxis, :] * rotMat_scene, axis=1 )
        lookat = np.sum((lookat * scale_scene)[np.newaxis, :] * rotMat_scene, axis=1 )
        up = np.sum((up * scale_scene )[np.newaxis, :] * rotMat_scene, axis=1 )

    if not trans_scene is None:
        origin = origin + trans_scene
        lookat = lookat + trans_scene

    up = up / np.sqrt(np.sum(up * up ) )

    return origin, lookat, up


def transformToXml(root ):
    rstring = et.tostring(root, 'utf-8')
    pstring = minidom.parseString(rstring)
    xmlString = pstring.toprettyxml(indent="    ")
    xmlString= xmlString.split('\n')
    xmlString = [x for x in xmlString if len(x.strip()) != 0 ]
    xmlString = '\n'.join(xmlString )
    return xmlString


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default="./xml1/", help="outDir of xml file" )
    parser.add_argument('--threshold', type=float, default = 0.3, help = 'the threshold to decide low quality mesh.' )
    parser.add_argument('--rs', type=int, default=0, help='the starting point' )
    parser.add_argument('--re', type=int, default=1600, help='the end point' )
    parser.add_argument('--sampleRate', type=float, default=100.0 )
    parser.add_argument('--sampleNum', type=int, default=3 )
    parser.add_argument('--heightMin', type=float, default=1.4 )
    parser.add_argument('--heightMax', type=float, default=1.8 )
    parser.add_argument('--distMin', type=float, default=0.3 )
    parser.add_argument('--distMax', type=float, default=1.5 )
    parser.add_argument('--thetaMin', type=float, default=-60 )
    parser.add_argument('--thetaMax', type=float, default=20 )
    parser.add_argument('--phiMin', type=float, default=-45 )
    parser.add_argument('--phiMax', type=float, default=45 )
    # Program
    parser.add_argument('--program', default='~/OptixRenderer/src/bin/optixRenderer' )

    opt = parser.parse_args()

    params = JSONHelper.read("./Parameters.json" )

    filename_json = params["scan2cad"]
    shapeNetRoot = params["shapenetAbs"]
    layoutRoot = params["scannet_layoutAbs"]
    camRootAbs = params['scannet_camAbs']

    sceneCnt = 0
    for r in JSONHelper.read(filename_json ):
        if not (sceneCnt >= opt.rs and sceneCnt < opt.re ):
            sceneCnt += 1
            continue
        sceneCnt += 1

        id_scan = r["id_scan"]


        outDir = osp.abspath(opt.out + "/" + id_scan )
        os.system('mkdir -p %s' % outDir )

        if not osp.isfile(osp.join(outDir, 'transform.dat') ):
            continue

        if osp.isfile(osp.join(outDir, 'cam.txt') ):
            continue

        # Load transformation file
        transformFile = osp.join(outDir, 'transform.dat' )
        with open(transformFile, 'rb') as fIn:
            transforms = pickle.load(fIn)

        # Sample initial camera pose
        camGap = int(opt.sampleRate / opt.sampleNum)

        poseDir = osp.join(camRootAbs, id_scan, 'pose')
        poseNum = len(glob.glob(osp.join(poseDir, '*.txt') ) )
        isSelected = np.zeros(poseNum, dtype=np.int32 )
        for n in range(0, poseNum, camGap ):
            isSelected[n] = 1

        samplePoint = int(poseNum / opt.sampleRate )

        camPoses= []
        for n in range(0, 10000, camGap ):
            poseFile = osp.join(poseDir, '%d.txt' % n)
            if not osp.isfile(poseFile ):
                break

            camMat = np.zeros((4, 4), dtype=np.float32 )

            isValidCam = True
            with open(poseFile, 'r') as camIn:
                for n in range(0, 4):
                    camLine = camIn.readline().strip()
                    if camLine.find('inf') != -1 or camLine.find('Inf') != -1:
                        isValidCam = False
                        break

                    camLine  = [float(x) for x in camLine.split(' ') ]
                    for m in range(0, 4):
                        camMat[n, m] = camLine[m]

            if isValidCam == False:
                while not isValidCam:
                    camMat = np.zeros((4,4), dtype=np.float32 )
                    while True:
                        camId = np.random.randint(0, poseNum )
                        if isSelected[camId ] == 0:
                            break
                    poseFile = osp.join(poseDir, '%d.txt' % camId )
                    isValidCam = True
                    with open(poseFile, 'r') as camIn:
                        for n in range(0, 4):
                            camLine = camIn.readline().strip()
                            if camLine.find('inf') != -1 or camLine.find('Inf') != -1:
                                isValidCam = False
                                break
                            camLine  = [float(x) for x in camLine.split(' ') ]

                            for m in range(0, 4):
                                camMat[n, m] = camLine[m]

                rot = camMat[0:3, 0:3]
                trans = camMat[0:3, 3]

                origin, lookat, up = computeCameraEx(rot, trans,
                        transforms[0][0][1], transforms[0][1][1], transforms[0][2][1] )
                isSelected[camId ] = 1

                origin = origin.reshape(1, 3 )
                lookat = lookat.reshape(1, 3 )
                up = up.reshape(1, 3 )
                camPose = np.concatenate([origin, lookat, up ], axis=0 )
                camPoses.append(camPose )
            else:
                rot = camMat[0:3, 0:3]
                trans = camMat[0:3, 3]

                origin, lookat, up = computeCameraEx(rot, trans,
                        transforms[0][0][1], transforms[0][1][1], transforms[0][2][1] )

                origin = origin.reshape(1, 3 )
                lookat = lookat.reshape(1, 3 )
                up = up.reshape(1, 3 )
                camPose = np.concatenate([origin, lookat, up ], axis=0 )
                camPoses.append(camPose )


        # Output the initial camera poses
        camNum = len(camPoses )
        with open(osp.join(outDir, 'camInitial.txt'), 'w') as camOut:
            camOut.write('%d\n' % camNum )
            print('Final sampled camera poses: %d' % len(camPoses ) )
            for camPose in camPoses:
                for n in range(0, 3):
                    camOut.write('%.3f %.3f %.3f\n' % \
                            (camPose[n, 0], camPose[n, 1], camPose[n, 2] ) )

        # Downsize the size of the image
        oldXML = osp.join(outDir, 'main.xml' )
        newXML = osp.join(outDir, 'mainTemp.xml')

        camFile = osp.join(outDir, 'camInitial.txt' )
        if not osp.isfile(oldXML ) or not osp.isfile(camFile ):
            continue

        tree = et.parse(oldXML )
        root  = tree.getroot()

        sensors = root.findall('sensor')
        for sensor in sensors:
            film = sensor.findall('film')[0]
            integers = film.findall('integer')
            for integer in integers:
                if integer.get('name' ) == 'width':
                    integer.set('value', '160')
                if integer.get('name' ) == 'height':
                    integer.set('value', '120')

        xmlString = transformToXml(root )
        with open(newXML, 'w') as xmlOut:
            xmlOut.write(xmlString )


        # Render depth and normal
        cmd = '%s -f %s -c %s -o %s -m %d' % (opt.program, newXML, 'camInitial.txt', 'im.rgbe', 2 )
        cmd += ' --forceOutput'
        os.system(cmd )

        cmd = '%s -f %s -c %s -o %s -m %d' % (opt.program, newXML, 'camInitial.txt', 'im.rgbe', 4 )
        cmd += ' --forceOutput'
        os.system(cmd )

        cmd = '%s -f %s -c %s -o %s -m %d' % (opt.program, newXML, 'camInitial.txt', 'im.rgbe', 5 )
        cmd += ' --forceOutput'
        os.system(cmd )

        # Load the normal and depth
        normalCosts = []
        depthCosts = []
        for n in range(0, camNum ):
            # Load the depth and normal
            normalName = osp.join(outDir, 'imnormal_%d.png' % (n+1) )
            maskName = osp.join(outDir, 'immask_%d.png' % (n+1) )
            depthName = osp.join(outDir, 'imdepth_%d.dat' % (n+1) )

            normal = cv2.imread(normalName )
            mask = cv2.imread(maskName )
            with open(depthName, 'rb') as fIn:
                hBuffer = fIn.read(4)
                height = struct.unpack('i', hBuffer)[0]
                wBuffer = fIn.read(4)
                width = struct.unpack('i', wBuffer)[0]
                dBuffer = fIn.read(4 * width * height )
                depth = np.asarray(struct.unpack('f' * height * width, dBuffer), dtype=np.float32 )
                depth = depth.reshape([height, width] )

            # Compute the ranking
            mask = (mask[:, :, 0] > 0.4 )
            mask = ndimage.binary_erosion(mask, border_value=1, structure=np.ones((3, 3) ) )
            mask = mask.astype(np.float32 )
            pixelNum = np.sum(mask )

            if pixelNum == 0:
                normalCosts.append(0 )
                depthCosts.append(0 )
                continue

            normal = normal.astype(np.float32 )
            normal_gradx = np.abs(normal[:, 1:] - normal[:, 0:-1] )
            normal_grady = np.abs(normal[1:, :] - normal[0:-1, :] )
            ncost = (np.sum(normal_gradx ) + np.sum(normal_grady ) ) / pixelNum

            dcost = np.sum(np.log(depth + 1 ) ) / pixelNum

            normalCosts.append(ncost )
            depthCosts.append(dcost )

        normalCosts = np.array(normalCosts, dtype=np.float32 )
        depthCosts = np.array(depthCosts, dtype=np.float32 )

        normalCosts = (normalCosts - normalCosts.min() ) \
                / (normalCosts.max() - normalCosts.min() )
        depthCosts = (depthCosts - depthCosts.min() ) \
                / (depthCosts.max() - depthCosts.min() )

        totalCosts = normalCosts + 0.3 * depthCosts

        camIndex = np.argsort(totalCosts )
        camIndex = camIndex[::-1]

        camPoses_s = []
        selectedDir = osp.join(outDir, 'selected' )
        if osp.isdir(selectedDir ):
            os.system('rm -r %s' % selectedDir )
        os.system('mkdir %s' % selectedDir )

        for n in range(0, min(samplePoint, camNum ) ):
            camPoses_s.append(camPoses[camIndex[n] ] )

            normalName = osp.join(outDir, 'imnormal_%d.png' % (camIndex[n] + 1) )
            os.system('cp %s %s' % (normalName, selectedDir ) )

        with open(osp.join(outDir, 'cam.txt'), 'w') as camOut:
            camOut.write('%d\n' % len(camPoses_s ) )
            print('Final sampled camera poses: %d' % len(camPoses_s ) )
            for camPose in camPoses_s:
                for n in range(0, 3):
                    camOut.write('%.3f %.3f %.3f\n' % \
                            (camPose[n, 0], camPose[n, 1], camPose[n, 2] ) )

        os.system('rm %s' % osp.join(outDir, 'mainTemp.xml') )
        os.system('rm %s' % osp.join(outDir, 'imnormal_*.png') )
        os.system('rm %s' % osp.join(outDir, 'immask_*.png') )
        os.system('rm %s' % osp.join(outDir, 'imdepth_*.dat') )
