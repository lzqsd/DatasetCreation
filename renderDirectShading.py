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
    parser.add_argument('--xmlFile', default="main")
    parser.add_argument('--outRoot', default='/siggraphasia20dataset/code/Routine/DatasetCreation/')
    parser.add_argument('--program', default='/siggraphasia20dataset/OptixRendererShading/src/bin/optixRenderer')
    parser.add_argument('--programNoOcclu', default='/siggraphasia20dataset/OptixRendererShadingNoOcclu/src/bin/optixRenderer')
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

        sceneId = outDir.split('/')[-1]
        xmlDir = osp.join(xmlRoot, sceneId )

        xmlFile = osp.join(xmlDir, '%s.xml' % opt.xmlFile )
        if not osp.isfile(xmlFile ):
            continue

        print('%d/%d: %s' % (k, min(opt.re, len(scenes ) ), sceneId ) )

        newFile_temp = osp.join(xmlDir, '%s-shading' % opt.xmlFile )

        tree = et.parse(xmlFile )
        root  = tree.getroot()

        shapeList = root.findall('shape' )
        lightList = []
        for shape in shapeList:
            shapeId = shape.get('id')
            if shapeId.find('window') != -1:
                # Change to the box
                lightList.append(shape )

            elif shapeId.find('ceiling_lamp') != -1:
                string = shape.findall('string')[0]
                filename = string.get('value')
                if filename.find('alignedNew.obj') != -1 \
                        or filename.find('aligned_light.obj') != -1:
                    emitters = shape.findall('emitter')
                    lightList.append(shape )

            elif shapeId.find('03636649') != -1:
                string = shape.findall('string')[0]
                filename = string.get('value')
                if filename.find('alignedNew.obj') != -1 \
                        or filename.find('aligned_light.obj') != -1:
                    lightList.append(shape )


        for n in range(0, len(lightList ) ):
            newFile = newFile_temp + '_%d.xml' % n
            light = lightList[n ]  

            # Create a new xml file
            newRoot = copy.deepcopy(root )
            newRoot.append(light )

            lightId = light.get('id')
            if shapeId.find('window') != -1:
                shapeList = newRoot.findall('shape' )
                for shape in shapeList:
                    emitters = shape.findall('emitter' )
                    for emitter in emitters:
                        shape.remove(emitter )
            else:
                emitters = newRoot.findall('emitter')
                for emitter in emitters:
                    newRoot.remove(emitter )
                shapeList = newRoot.findall('shape' )
                for shape in shapeList:
                    shapeId = shape.get('id')
                    if shapeId != lightId:
                        emitters = shape.findall('emitter' )
                        for emitter in emitters:
                            shape.remove(emitter )


            xmlString = transformToXml(newRoot )
            with open(newFile, 'w') as xmlOut:
                xmlOut.write(xmlString )
            
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
            
            center = np.mean(bverts, axis=0 )

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

            xAxis = xAxis * xLen
            yAxis = yAxis * yLen

            cmd = '%s -c cam.txt -f %s -o %s -m 0 --center %.4f %.4f %.4 ' \
                    + '--yAxis %.3f %.3f %.3f --xAxis %.3f %.3f %.3f ' \
                    + '--zAxis %.3f %.3f %.3f'
                    % (opt.program, newFile, osp.join(outDir, 'imDS%d.rgbe' % n ), 
                            center[0], center[1], center[2], 
                            yAxis[0], yAxis[1], yAxis[2], 
                            xAxis[0], xAxis[1], xAxis[2], 
                            zAxis[0], zAxis[1], zAxis[2] ) 
            os.system(cmd )
            cmd = '%s -c cam.txt -f %s -o %s -m 0 --center %.4f %.4f %.4 ' \
                    + '--yAxis %.3f %.3f %.3f --xAxis %.3f %.3f %.3f ' \
                    + '--zAxis %.3f %.3f %.3f'
                    % (opt.programNoOcclu, newFile, osp.join(outDir, 'imDSNoOcclu%d.rgbe' % n ), 
                            center[0], center[1], center[2], 
                            yAxis[0], yAxis[1], yAxis[2], 
                            xAxis[0], xAxis[1], xAxis[2], 
                            zAxis[0], zAxis[1], zAxis[2] ) 
            os.system(cmd ) 
            
            DSNames = glob.glob(osp.join(outDir, 'imDS%d_*.rgbe') )


            DSNoNames = glob.glob(osp.join(outDir, 'imDSNoOcclu%d*.rgbe' % n )
                    )
