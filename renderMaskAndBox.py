import numpy as np
import os
import shutil
import glob
import argparse
import os.path as osp
import xml.etree.ElementTree as et
from xml.dom import minidom
import pickle
import random
import copy
import cv2
import pickle

def loadMesh(name ):
    vertices = []
    faces = []
    with open(name, 'r') as meshIn:
        lines = meshIn.readlines()
    lines = [x.strip() for x in lines if len(x.strip() ) > 2 ]
    for l in lines:
        if l[0:2] == 'v ':
            vstr = l.split(' ')[1:4]
            varr = [float(x) for x in vstr ]
            varr = np.array(varr ).reshape([1, 3] )
            vertices.append(varr )
        elif l[0:2] == 'f ':
            fstr = l.split(' ')[1:4]
            farr = [int(x.split('/')[0] ) for x in fstr ]
            farr = np.array(farr ).reshape([1, 3] )
            faces.append(farr )

    vertices = np.concatenate(vertices, axis=0 )
    faces = np.concatenate(faces, axis=0 )
    return vertices, faces


def writeMesh(name, vertices, faces ):
    with open(name, 'w') as meshOut:
        for n in range(0, vertices.shape[0]):
            meshOut.write('v %.3f %.3f %.3f\n' %
                    (vertices[n, 0], vertices[n, 1], vertices[n, 2] ) )
        for n in range(0,faces.shape[0] ):
            meshOut.write('f %d %d %d\n' %
                    (faces[n, 0], faces[n, 1], faces[n, 2]) )


def computeBox(vertices ):
    minX, maxX = vertices[:, 0].min(), vertices[:, 0].max()
    minY, maxY = vertices[:, 1].min(), vertices[:, 1].max()
    minZ, maxZ = vertices[:, 2].min(), vertices[:, 2].max()

    corners = []
    corners.append(np.array([minX, minY, minZ] ).reshape(1, 3) )
    corners.append(np.array([maxX, minY, minZ] ).reshape(1, 3) )
    corners.append(np.array([maxX, minY, maxZ] ).reshape(1, 3) )
    corners.append(np.array([minX, minY, maxZ] ).reshape(1, 3) )

    corners.append(np.array([minX, maxY, minZ] ).reshape(1, 3) )
    corners.append(np.array([maxX, maxY, minZ] ).reshape(1, 3) )
    corners.append(np.array([maxX, maxY, maxZ] ).reshape(1, 3) )
    corners.append(np.array([minX, maxY, maxZ] ).reshape(1, 3) )

    corners = np.concatenate(corners ).astype(np.float32 )

    faces = []
    faces.append(np.array([1, 2, 3] ).reshape(1, 3) )
    faces.append(np.array([1, 3, 4] ).reshape(1, 3) )

    faces.append(np.array([5, 7, 6] ).reshape(1, 3) )
    faces.append(np.array([5, 8, 7] ).reshape(1, 3) )

    faces.append(np.array([1, 6, 2] ).reshape(1, 3) )
    faces.append(np.array([1, 5, 6] ).reshape(1, 3) )

    faces.append(np.array([2, 7, 3] ).reshape(1, 3) )
    faces.append(np.array([2, 6, 7] ).reshape(1, 3) )

    faces.append(np.array([3, 8, 4] ).reshape(1, 3) )
    faces.append(np.array([3, 7, 8] ).reshape(1, 3) )

    faces.append(np.array([4, 5, 1] ).reshape(1, 3) )
    faces.append(np.array([4, 8, 5] ).reshape(1, 3) )

    faces = np.concatenate(faces ).astype(np.int32 )

    return corners, faces

def transformToXml(root ):
    rstring = et.tostring(root, 'utf-8')
    pstring = minidom.parseString(rstring)
    xmlString = pstring.toprettyxml(indent="    ")
    xmlString= xmlString.split('\n')
    xmlString = [x for x in xmlString if len(x.strip()) != 0 ]
    xmlString = '\n'.join(xmlString )
    return xmlString

def computeRotMat(angle, axis ):
    axis = axis / np.sqrt(np.sum(axis * axis ) )
    w = np.zeros( (3, 3) )
    w[0, 1], w[0, 2] = -axis[2], axis[1]
    w[1, 0], w[1, 2] = axis[2], -axis[0]
    w[2, 0], w[2, 1] = -axis[1], axis[0]
    angle = (angle / 180.0) * np.pi
    axis = axis.reshape(3, 1)
    axis_t= axis.reshape(1, 3)

    rotMat = np.sin(angle ) * w \
            + np.cos(angle ) * (np.eye(3) - np.matmul(axis, axis_t ) ) \
            + np.matmul(axis, axis_t )
    return rotMat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Directories
    parser.add_argument('--xmlRoot', default='/siggraphasia20dataset/code/Routine/scenes/xml')
    parser.add_argument('--xmlFile', default="main" )
    parser.add_argument('--outRoot', default='/siggraphasia20dataset/code/Routine/DatasetCreation/')
    parser.add_argument('--program', default='/siggraphasia20dataset/OptixRenderer/src/bin/optixRenderer')
    # Start and end point
    parser.add_argument('--rs', default=0, type=int, help='the width of the image' )
    parser.add_argument('--re', default=1600, type=int, help='the height of the image' )
    opt = parser.parse_args()

    xmlFile = opt.xmlFile
    xmlRoot = opt.xmlRoot

    outRoot = xmlFile.split('/')[-1] + '_' + xmlRoot.split('/')[-1]
    outRoot = osp.join(opt.outRoot, outRoot )
    scenes = glob.glob(osp.join(outRoot, 'scene*' ) )
    scenes = sorted(scenes )

    shapeRoot = '/siggraphasia20dataset/uv_mapped/'

    for k in range(opt.rs, min(opt.re, len(scenes ) ) ):
        outDir = scenes[k]
        maskDir = outDir.replace('mainDiffLight', 'main')
        maskDir = maskDir.replace('mainDiffMat', 'main')

        sceneId = outDir.split('/')[-1]
        xmlDir = osp.join(xmlRoot, sceneId )

        xmlFile = osp.join(xmlDir, '%s.xml' % opt.xmlFile )
        if not osp.isfile(xmlFile ):
            continue

        print('%d/%d: %s' % (k, min(opt.re, len(scenes ) ), sceneId ) )

        newFile_temp = osp.join(xmlDir, '%s-light' % opt.xmlFile )

        tree = et.parse(xmlFile )
        root  = tree.getroot()

        shapeList = root.findall('shape' )
        lightList = []
        for shape in shapeList:
            shapeId = shape.get('id')
            if shapeId.find('window') != -1:
                # Change to the box
                string = shape.findall('string')[0]
                filename = string.get('value')
                filename = filename.replace('alignedNew', 'alignedNew_box')
                string.set('value', filename )

                refs = shape.findall('ref')
                for ref in refs:
                    shape.remove(ref )
                lightList.append(shape )

            elif shapeId.find('ceiling_lamp') != -1:
                string = shape.findall('string')[0]
                filename = string.get('value')
                if filename.find('alignedNew.obj') != -1 \
                        or filename.find('aligned_light.obj') != -1:
                    emitters = shape.findall('emitter')
                    for emitter in emitters:
                        shape.remove(emitter )

                    refs = shape.findall('ref')
                    for ref in refs:
                        shape.remove(ref )
                    lightList.append(shape )

            elif shapeId.find('03636649') != -1:
                string = shape.findall('string')[0]
                filename = string.get('value')
                if filename.find('alignedNew.obj') != -1 \
                        or filename.find('aligned_light.obj') != -1:
                    emitters = shape.findall('emitter')
                    for emitter in emitters:
                        shape.remove(emitter )

                    refs = shape.findall('ref')
                    for ref in refs:
                        shape.remove(ref )
                    lightList.append(shape )

        for shape in shapeList:
            root.remove(shape )

        emitters = root.findall('emitter')
        for emitter in emitters:
            root.remove(emitter )

        # Check if masks have been rendered
        maskNames = glob.glob(osp.join(maskDir, 'immask_*.png' ) )
        if len(maskNames ) == 0:
            cmd = '%s -c cam.txt -f %s -o %s -m 4' \
                    % (opt.program, xmlFile, osp.join(maskDir, 'im.rgbe' ) )
            os.system(cmd )


        # Load the camera
        origins, ups, targets = [], [], []
        camFIle = osp.join(xmlDir, 'cam.txt' )
        with open(camFIle, 'r' ) as camIn:
            camNum = int(camIn.readline().strip() )
            for m in range(0, camNum ):
                origArr = camIn.readline().strip()
                origArr = [float(x) for x in origArr.split(' ') ]
                origins.append(np.array(origArr ).reshape(1, 3) )

                targetArr = camIn.readline().strip()
                targetArr = [float(x) for x in targetArr.split(' ') ]
                targets.append(np.array(targetArr ).reshape(1, 3) )

                upArr = camIn.readline().strip()
                upArr = [float(x) for x in upArr.split(' ') ]
                ups.append(np.array(upArr ).reshape(1, 3) )

        origins = np.concatenate(origins, axis=0 )
        targets = np.concatenate(targets, axis=0 )
        ups = np.concatenate(ups, axis=0 )


        for n in range(0, len(lightList ) ):
            newFile = newFile_temp + '_%d.xml' % n
            light = lightList[n ]
            # Create a new xml file
            newRoot = copy.deepcopy(root )
            newRoot.append(light )

            xmlString = transformToXml(newRoot )
            with open(newFile, 'w') as xmlOut:
                xmlOut.write(xmlString )

            isWindow = False
            lightId = light.get('id')
            if lightId.find('window') != -1:
                isWindow = True

            cmd = '%s -c cam.txt -f %s -o %s -m 4' \
                    % (opt.program, newFile, osp.join(outDir, 'light%d.rgbe' % n ) )
            os.system(cmd )

            # Compute the 3D bounding box
            string = light.findall('string')[0]
            filename = string.get('value')
            filename = '/'.join(filename.split('/')[-3:] )
            filename = osp.join(shapeRoot, filename )
            vertices, faces = loadMesh(filename )
            bverts, bfaces = computeBox(vertices )

            transform = light.findall('transform')[0]
            for trans in transform:
                tag = trans.tag
                if tag == 'scale':
                    x = float(trans.get('x') )
                    y = float(trans.get('y') )
                    z = float(trans.get('z') )
                    scale = np.array([x, y, z] ).reshape(1, 3)
                    bverts = bverts * scale
                elif tag == 'rotate':
                    x = float(trans.get('x') )
                    y = float(trans.get('y') )
                    z = float(trans.get('z') )
                    angle = float(trans.get('angle') )
                    axis = np.array([x, y, z] )
                    rotMat = computeRotMat(angle, axis )
                    bverts = np.matmul(rotMat, bverts.transpose() )
                    bverts = bverts.transpose()

                elif tag == 'translate':
                    x = float(trans.get('x') )
                    y = float(trans.get('y') )
                    z = float(trans.get('z') )
                    translate = np.array([x, y, z] ).reshape(1, 3 )
                    bverts = bverts + translate

            boxObjName = osp.join(xmlDir, 'light%d.obj' % n )
            writeMesh(boxObjName, bverts, bfaces )

            axis = []
            axis.append(bverts[1, :] - bverts[0, :] )
            axis.append(bverts[2, :] - bverts[1, :] )
            axis.append(bverts[4, :] - bverts[0, :] )
            axisLen = []
            for m in range(0, 3):
                l = np.sqrt(np.sum(axis[m] * axis[m] ) )
                axisLen.append(l )
                axis[m] = axis[m] / l

            yLen = [abs(axis[0][1] ), abs(axis[1][1] ), abs(axis[2][1] ) ]
            if yLen[0]>= yLen[1] and yLen[0] >= yLen[2]:
                yId = 0
                if axisLen[2] >= axisLen[1]:
                    xId = 2
                    zId = 1
                else:
                    xId = 1
                    zId = 2
            elif yLen[1] >= yLen[0] and yLen[1] >= yLen[2]:
                yId = 1
                if axisLen[0] >= axisLen[2]:
                    xId = 0
                    zId = 2
                else:
                    xId = 2
                    zId = 0
            elif yLen[2] >= yLen[0] and yLen[2] >= yLen[1]:
                yId = 2
                if axisLen[1] >= axisLen[0]:
                    xId = 1
                    zId = 0
                else:
                    xId = 0
                    zId = 1

            xAxis = axis[xId ]
            yAxis = axis[yId ]

            yLen = axisLen[yId ]
            xLen = axisLen[xId ]
            zLen = axisLen[zId ]

            if yAxis[1] < 0:
                yAxis = -yAxis
            zAxis = np.cross(xAxis, yAxis )


            # Compute the 2D bounding box
            for m in range(1, camNum + 1 ):
                origin = origins[m-1]
                target = targets[m-1]
                up = ups[m-1]

                lightMaskName = osp.join(outDir, 'light%dmask_%d.png' % (n, m ) )
                maskName = lightMaskName.replace(outDir, \
                        maskDir ).replace('light%d' % n, 'im')
                assert(osp.isfile(maskName ) )

                lightMask = cv2.imread(lightMaskName )
                mask = cv2.imread(maskName )

                if isWindow:
                    lightMask = np.logical_and(mask < 5, lightMask > 250 )
                    lightMask = lightMask.astype(np.uint8 ) * 255

                # Compute the 2D bounding box
                lmask_col = np.sum(lightMask.astype(np.int32 ), axis=0 )
                lmask_col = np.sum(lmask_col, axis=1 )
                if np.sum(lmask_col ) > 0:
                    lmask_col_ind = np.argwhere(lmask_col > 0 )
                    x1, x2 = np.min(lmask_col_ind), np.max(lmask_col_ind )

                    lmask_row = np.sum(lightMask.astype(np.int32 ), axis=1 )
                    lmask_row = np.sum(lmask_row, axis=1 )
                    lmask_row_ind = np.argwhere(lmask_row > 0 )
                    y1, y2 = np.min(lmask_row_ind), np.max(lmask_row_ind )
                else:
                    x1, x2, y1, y2 = None, None, None, None

                # Compute the 3D bounding box
                cam_zAxis = origin - target
                cam_zAxis = cam_zAxis / np.sqrt(np.sum(cam_zAxis * cam_zAxis ) )
                cam_yAxis = up 
                cam_xAxis = np.cross(cam_yAxis, cam_zAxis )
                cam_rotMat = np.concatenate([
                    cam_xAxis[np.newaxis, :],
                    cam_yAxis[np.newaxis, :],
                    cam_zAxis[np.newaxis, :] ], axis=0 )

                bverts_cam = bverts - origin.reshape(1,3 )
                bverts_cam_center = np.mean(bverts_cam, axis=0 )
                cam_y = np.matmul(cam_rotMat, yAxis.reshape(3, 1) ).squeeze()
                cam_x = np.matmul(cam_rotMat, xAxis.reshape(3, 1) ).squeeze()
                cam_z = np.matmul(cam_rotMat, zAxis.reshape(3, 1) ).squeeze()
                bverts_cam_center = np.matmul(cam_rotMat, bverts_cam_center.reshape(3, 1) ).squeeze()

                if np.sum(cam_z * bverts_cam_center ) > 0:
                    cam_z = -cam_z
                    cam_x = -cam_x

                box3D = {'center': bverts_cam_center,
                        'xAxis': cam_x, 'yAxis': cam_y, 'zAxis' : cam_z,
                        'xLen': xLen, 'yLen': yLen, 'zLen' : zLen }
                box2D = {'x1': x1, 'x2': x2, 'y1' : y1, 'y2' : y2 }
                boxDic = {'isWindow': isWindow, 'box3D': box3D, 'box2D' : box2D }

                # Create a light directory
                lightDir = lightMaskName.replace('%dmask' % n, '').replace('.png', '')
                if not osp.isdir(lightDir ):
                    os.system('mkdir %s' % lightDir )

                newLightMaskName = osp.join(lightDir, 'mask%d.png' % n )
                cv2.imwrite(newLightMaskName, lightMask )

                boxFileName = osp.join(lightDir, 'box%d.dat' % n )
                with open(boxFileName, 'wb') as fOut:
                    pickle.dump(boxDic, fOut )

                os.system('rm %s' % lightMaskName )
