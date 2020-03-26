import numpy as np
import os
import shutil
import glob
import JSONHelper
import quaternion
import argparse
import os.path as osp
import xml.etree.ElementTree as et
from xml.dom import minidom
import pickle
import align_utils as utils
import random

matCatList = ['wood', 'plastic', 'rough_stone', 'specular_stone', 'paint', 'leather', 'metal', 'fabric', 'rubber']


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Directories
    parser.add_argument('--out', default="./xml1/", help="outdir of xml file" )
    parser.add_argument('--annotation', default='/newfoundland/zhl/Scan2cad/full_annotations.json', help='the file of the annotation' )
    # Rendering parameters
    parser.add_argument('--width', default=640, type=int, help='the width of the image' )
    parser.add_argument('--height', default=480, type=int, help='the height of the image' )
    parser.add_argument('--camGap', default=100, type=int, help='the gap to sample camera positions' )
    parser.add_argument('--sampleCount', default=1024, type=int, help='the by default number of samples' )
    # Material lists
    parser.add_argument('--matList', default='./MatLists/', help='the list of materials for objects' )
    parser.add_argument('--sceneMatList', default='./MatSceneLists/', help='the list of materials for scenes' )
    # Lighting parameters
    parser.add_argument('--envScaleMean', default=80, type=float, help='the mean of envScale' )
    parser.add_argument('--envScaleStd', default=40, type=float, help='the std of envScale' )
    # Start and end point
    parser.add_argument('--rs', default=0, type=int, help='the width of the image' )
    parser.add_argument('--re', default=1600, type=int, help='the height of the image' )
    opt = parser.parse_args()

    params = JSONHelper.read("./Parameters.json")

    filename_json = opt.annotation

    shapeNetRoot = params["shapenet"]
    layoutRoot = params["scannet_layout"]
    adobeRoot = params['adobestock']
    envRoot = params['envmap']

    shapeNetRootAbs = params['shapenetAbs']
    layoutRootAbs = params["scannet_layoutAbs"]
    camRootAbs = params['scannet_camAbs']
    adobeRootAbs = params['adobestockAbs']
    envRootAbs = params['envmapAbs']

    sceneCnt = 0
    camCnt = 0
    for r in JSONHelper.read(filename_json ):
        if not(sceneCnt >= opt.rs and sceneCnt < opt.re):
            continue
        sceneCnt += 1

        id_scan = r["id_scan"]

        outdir = osp.join(opt.out, id_scan)
        camOutFile = osp.join(outdir, 'cam.txt')
        xmlOutFile = osp.join(outdir, 'main.xml')
        transformFile = osp.join(outdir, 'transform.dat')
        if not osp.isfile(transformFile ):
            continue


        camDir = 'xml1Cam'
        with open(camOutFile, 'r') as camIn:
            lines = camIn.readlines()
        isFindNan = False
        for l in lines:
            if l.find('nan') != -1:
                isFindNan = True
                break

        if isFindNan == True:
            camCnt += 1
            print(camCnt, id_scan )
        else:
            continue

        outdirNew = osp.join('xml1Cam', id_scan )
        os.system('mkdir -p %s' % outdirNew )

        # load transformations
        with open(transformFile, 'rb') as fIn:
            transforms = pickle.load(fIn)

        camOutFileNew = osp.join(outdirNew, 'cam.txt')

        ############################################################################################
        # Write camera pose
        with open(camOutFileNew, 'w') as camOut:
            poseDir = osp.join(osp.join(camRootAbs, id_scan), 'pose')

            poseNum = len(glob.glob(osp.join(poseDir, '*.txt') ) )
            isSelected = np.zeros(poseNum, dtype=np.int32 )
            for n in range(int(opt.camGap/2), poseNum, opt.camGap ):
                isSelected[n] = 1

            camMats = []
            for n in range(int(opt.camGap/2), 10000, opt.camGap ):
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

                    camMats.append(camMat )
                    isSelected[camId ] = 1
                else:
                    camMats.append(camMat )


            camOut.write('%d\n' % len(camMats ) )
            for camMat in camMats:
                rot = camMat[0:3, 0:3]
                trans = camMat[0:3, 3]

                origin, lookat, up = computeCameraEx(rot, trans,
                        transforms[0][0][1], transforms[0][1][1], transforms[0][2][1] )
                camOut.write('%.3f %.3f %.3f\n' % (origin[0], origin[1], origin[2] ) )
                camOut.write('%.3f %.3f %.3f\n' % (lookat[0], lookat[1], lookat[2] ) )
                camOut.write('%.3f %.3f %.3f\n' % (up[0], up[1], up[2] ) )
