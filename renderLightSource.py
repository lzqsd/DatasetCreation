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
import SGOptim

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


def compareTransforms(trans1, trans2 ):
    trans1 = trans1.findall('./*')
    trans2 = trans2.findall('./*')

    if len(trans1 ) != len(trans2 ):
        return False

    for n in range(0, len(trans1) ):
        tr1 = trans1[n]
        tr2 = trans2[n]
        if tr1.tag != tr2.tag:
            return False

        if tr1.tag == 'scale':
            if(tr1.get('x') != tr2.get('x') ):
                return False
            if(tr1.get('y') != tr2.get('y') ):
                return False
            if(tr1.get('z') != tr2.get('z') ):
                return False

        elif tr1.tag == 'rotate':
            if(tr1.get('x') != tr2.get('x') ):
                return False
            if(tr1.get('y') != tr2.get('y') ):
                return False
            if(tr1.get('z') != tr2.get('z') ):
                return False
            if(tr1.get('angle') != tr2.get('angle') ):
                return False

        elif tr1.tag == 'translate':
            if(tr1.get('x') != tr2.get('x') ):
                return False
            if(tr1.get('y') != tr2.get('y') ):
                return False
            if(tr1.get('z') != tr2.get('z') ):
                return False
        else:
            assert(False )

    return True


def initializeEnv(imOrig ):
    im = np.sum(imOrig, axis=2 )
    height, width = im.shape
    im = im.flatten()
    imId = np.argmax(im )
    rowId = int(imId / width )
    colId = imId - rowId * width

    #print(rowId, colId )

    # The weight
    weight = imOrig[rowId, colId, :]
    # The theta
    theta = (rowId + 0.5) / height * np.pi
    phi = ((colId + 0.5 ) / width - 0.5) * np.pi * 2

    return weight, theta, phi


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Directories
    parser.add_argument('--xmlRoot',
            default='/siggraphasia20dataset/code/Routine/scenes/xml1')
    parser.add_argument('--xmlFile', default="main" )
    parser.add_argument('--outRoot', default='/siggraphasia20dataset/code/Routine/DatasetCreation/')
    parser.add_argument('--program', default='/siggraphasia20dataset/OptixRenderer/src/bin/optixRenderer')
    # Start and end point
    parser.add_argument('--rs', default=3, type=int, help='the width of the image' )
    parser.add_argument('--re', default=4, type=int, help='the height of the image' )
    opt = parser.parse_args()

    xmlFile = opt.xmlFile
    xmlRoot = opt.xmlRoot

    outRoot = xmlFile.split('/')[-1] + '_' + xmlRoot.split('/')[-1]
    outRoot = osp.join(opt.outRoot, outRoot )
    scenes = glob.glob(osp.join(outRoot, 'scene*' ) )
    scenes = sorted(scenes )

    for k in range(opt.rs, min(opt.re, len(scenes ) ) ):
        outDir = scenes[k]

        sceneId = outDir.split('/')[-1]
        xmlDir = osp.join(xmlRoot, sceneId )

        xmlFile = osp.join(xmlDir, '%s.xml' % opt.xmlFile )
        if not osp.isfile(xmlFile ):
            continue

        print('%d/%d: %s' % (k, min(opt.re, len(scenes ) ), sceneId ) )

        # Load the original xml file
        tree = et.parse(xmlFile )
        root  = tree.getroot()
        emitter = root.findall('emitter')[0]
        scale = emitter.findall('float')[0]
        scale = float(scale.get('value') )
        if scale < 0.01:
            isEnv = False
        else:
            isEnv = True

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

        # Start
        newFiles = glob.glob(osp.join(xmlDir, 'main-light_*.xml') )
        for newFile in newFiles:
            lightId = int(newFile.split('_')[-1].split('.')[0] )
            # Load the box xml file
            boxTree = et.parse(newFile )
            boxRoot = boxTree.getroot()
            shape = boxRoot.findall('shape')[0]
            shapeId = shape.get('id')
            transform = shape.findall('transform')[0]

            string = shape.findall('string')[0]
            filename = string.get('value')

            if shapeId.find('window') != -1:
                isWindow = True
            else:
                isWindow = False

            if isEnv and isWindow:
                # Get the location
                boxDir = outDir.replace('mainDiffLight', 'main').replace('mainDiffMat', 'main')
                boxFile = osp.join(boxDir, 'light_1', 'box%d.dat' % lightId )

                with open(boxFile, 'rb') as fin:
                    boxData = pickle.load(fin )
                assert(boxData['isWindow'] == True )

                center_cam = boxData['box3D']['center']
                zAxis_cam = boxData['box3D']['zAxis']
                zLen = boxData['box3D']['zLen']

                # Compute the camera pose
                origin = origins[0]
                target = targets[0]
                up = ups[0]

                cam_zAxis = origin - target
                cam_zAxis = cam_zAxis / np.sqrt(np.sum(cam_zAxis * cam_zAxis ) )
                cam_yAxis = up
                cam_xAxis = np.cross(cam_yAxis, cam_zAxis )
                cam_rotMat = np.concatenate([
                    cam_xAxis[:, np.newaxis],
                    cam_yAxis[:, np.newaxis],
                    cam_zAxis[:, np.newaxis] ], axis=1 )
                zAxis = np.matmul(cam_rotMat, zAxis_cam.reshape(3, 1) ).squeeze()
                center = np.matmul(cam_rotMat, center_cam.reshape(3, 1) ).squeeze() + origin

                '''
                objFile = osp.join(xmlDir, 'light%d.obj' % lightId )
                bverts, bfaces = loadMesh(objFile )
                bcenter = np.mean(bverts, axis=0 )

                print(bcenter, center )
                print( (bverts[1] - bverts[0] ) / np.linalg.norm(bverts[1] - bverts[0] ) )
                print( (bverts[2] - bverts[1] ) / np.linalg.norm(bverts[2] - bverts[1] ) )
                print( (bverts[4] - bverts[0] ) / np.linalg.norm(bverts[4] - bverts[0] ) )
                print(zAxis )
                '''


                # Create new xml files, remove windows, change the sensor
                lightRoot = copy.deepcopy(root )

                lightShapes = lightRoot.findall('shape')
                for lightShape in lightShapes:
                    emitters = lightShape.findall('emitter')
                    if len(emitters ) != 0:
                        lightShape.remove(emitters[0] )
                    lightShapeId = lightShape.get('id')
                    if lightShapeId.find('window') != -1:
                        lightRoot.remove(lightShape )

                sensor = lightRoot.findall('sensor')[0]
                sensor.set('type', 'envmap')
                film = sensor.findall('film')[0]
                integers = film.findall('integer')
                for integer in integers:
                    if integer.get('name') == 'width':
                        integer.set('value', '%d' % 512 )
                    elif integer.get('name') == 'height':
                        integer.set('value', '%d' % 256 )

                camTransform = et.SubElement(sensor, 'transform')
                camTransform.set('type', 'toWorld')
                lookAt = et.SubElement(camTransform, 'lookAt')
                lookAt.set('origin', '%.3f %.3f %.3f' \
                        % (center[0], center[1], center[2] ) )
                lookAt.set('target', '%.3f %.3f %.3f' \
                        % (center[0] + zAxis[0], center[1] + zAxis[1], center[2] + zAxis[2] ) )
                lookAt.set('up', '0 1 0')

                lightXmlFile = newFile.replace('main', opt.xmlFile ).replace('-light', '-lightSource')
                xmlString = transformToXml(lightRoot )
                with open(lightXmlFile, 'w') as xmlOut:
                    xmlOut.write(xmlString )

                # Render the panoroma
                cmd = '%s -f %s -o %s-light%d.rgbe --maxPathLength 1 --forceOutput' \
                        % (opt.program, lightXmlFile, opt.xmlFile, lightId )
                os.system(cmd )

                # Get the spherical Gaussain Parameters
                imName = osp.join(xmlDir, '%s-light%d_1.rgbe' % (opt.xmlFile, lightId ) )
                im = cv2.imread(imName, -1 )
                im = np.ascontiguousarray(im[:, :, ::-1] )
                assert(im is not None )
                assert(im.shape[0] == 256 )
                assert(im.shape[1] == 512 )

                # Run the optimization
                im[0:128, :, :] = 0 ## Should not do that, need to debug
                weight, theta, phi = initializeEnv(im )
                envOptim = SGOptim.SGEnvOptim(
                        weightValue = weight,
                        thetaValue = theta,
                        phiValue = phi )

                envmap = im.transpose([2, 0, 1] )[np.newaxis, :]
                theta, phi, lamb, weight, recIm = envOptim.optimizeAdam(envmap )

                del envOptim

                theta = theta.squeeze()
                phi = phi.squeeze()
                intensity = weight.squeeze()
                recIm = recIm.squeeze()

                recIm = recIm.transpose([1, 2, 0] )
                cv2.imwrite(imName.replace('.rgbe', 'rec.hdr'), recIm[:, :, ::-1] )

                # From theta and phi to axis
                envAxis_z = zAxis
                envAxis_z = envAxis_z / np.linalg.norm(envAxis_z )
                envAxis_y = np.array([0, 1, 0] ).astype(np.float32 )
                envAxis_x = np.cross(envAxis_z, envAxis_y )
                envAxis_x = envAxis_x / np.linalg.norm(envAxis_x )
                envAxis_y = np.cross(envAxis_x, envAxis_z )
                axis = np.sin(theta ) * np.cos(phi ) * envAxis_x \
                        + np.sin(theta ) * np.sin(phi ) * envAxis_y \
                        + np.cos(theta ) * envAxis_z

                # Change the coordinate, save the results
                for n in range(0, camNum ):
                    origin = origins[n]
                    target = targets[n]
                    up = ups[n]

                    # Compute the camera matrix
                    cam_zAxis = origin - target
                    cam_zAxis = cam_zAxis / np.sqrt(np.sum(cam_zAxis * cam_zAxis ) )
                    cam_yAxis = up
                    cam_xAxis = np.cross(cam_yAxis, cam_zAxis )
                    cam_rotMat = np.concatenate([
                        cam_xAxis[np.newaxis, :],
                        cam_yAxis[np.newaxis, :],
                        cam_zAxis[np.newaxis, :] ], axis=0 )

                    axis_cam = np.matmul(cam_rotMat, axis.reshape(3, 1) ).squeeze()
                    lightSource = {'intensity' : intensity, \
                            'lamb' : lamb, 'axis': axis_cam }
                    lightDir = osp.join(outDir, 'light_%d' % (n + 1) )
                    if not osp.isdir(lightDir ):
                        os.system('mkdir %s' % lightDir )
                    lightFile = osp.join(lightDir, 'light%d.dat' % lightId )
                    with open(lightFile, 'wb') as fout:
                        pickle.dump(lightSource, fout )

            elif not isEnv and isWindow:
                intensity = np.array([0.0, 0.0, 0.0] ).astype(np.float32 )
                lamb = 0.0
                axis = np.array([0.0, 1.0, 0.0] ).astype(np.float32 )
                lightSource = {'intensity' : intensity, \
                        'lamb' : lamb, 'axis': axis }

                # Save the results
                for n in range(0, camNum ):
                    lightDir = osp.join(outDir, 'light_%d' % (n + 1) )
                    if not osp.isdir(lightDir ):
                        os.system('mkdir %s' % lightDir )
                    lightFile = osp.join(lightDir, 'light%d.dat' % lightId )
                    with open(lightFile, 'wb') as fout:
                        pickle.dump(lightSource, fout )

            elif not isWindow:
                # Get the intensity
                intensity = None
                origShapes = root.findall('shape')
                for origShape in origShapes:
                    origShapeId = origShape.get('id')
                    if origShapeId == shapeId:
                        origString = origShape.findall('string')[0]
                        origFilename = origString.get('value')
                        if origFilename == filename:
                            origTransform = origShape.findall('transform')[0]
                            if compareTransforms(origTransform, transform ) == True:
                                assert(intensity is None )
                                emitters = origShape.findall('emitter')
                                if len(emitters ) == 1:
                                    emitter = emitters[0]
                                    rgb = emitter.findall('rgb')[0]
                                    intArr = rgb.get('value')
                                    intArr = [float(x) for x in intArr.split(' ') ]
                                    intensity = np.array(intArr ).astype(np.float32 )
                                elif len(emitters ) == 0:
                                    intensity = np.array([0.0, 0.0, 0.0]).astype(np.float32 )
                                else:
                                    assert(False )

                assert(intensity is not None )
                lamb = 0.0
                axis = np.array([0.0, 1.0, 0.0] ).astype(np.float32 )
                lightSource = {'intensity' : intensity, \
                        'lamb': lamb, 'axis' : axis }

                # Save the results
                for n in range(0, camNum ):
                    lightDir = osp.join(outDir, 'light_%d' % (n + 1) )
                    if not osp.isdir(lightDir ):
                        os.system('mkdir %s' % lightDir )
                    lightFile = osp.join(lightDir, 'light%d.dat' % lightId )
                    with open(lightFile, 'wb') as fout:
                        pickle.dump(lightSource, fout )

