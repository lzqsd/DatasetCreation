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


def addShape(root, name, fileName, transforms = None, materials = None ):
    shape = et.SubElement(root, 'shape')
    shape.set('id', '%s_object' % name )
    shape.set('type', 'obj' )

    stringF = et.SubElement(shape, 'string' )
    stringF.set('name', 'filename' )
    stringF.set('value', fileName )

    if transforms != None:
        transform = et.SubElement(shape, 'transform')
        transform.set('name', 'toWorld')
        for tr in transforms:
            if tr[0] == 's':
                s = tr[1]
                scale = et.SubElement(transform, 'scale')
                scale.set('x', '%.6f' % s[0] )
                scale.set('y', '%.6f' % s[1] )
                scale.set('z', '%.6f' % s[2] )

            elif tr[0] == 'rot':
                rotMat = tr[1]
                rotTr = rotMat[0,0] + rotMat[1,1] + rotMat[2,2]
                rotCos = np.clip((rotTr - 1) * 0.5, -1, 1 )
                rotAngle = np.arccos(np.clip(rotCos, -1, 1 ) )
                if np.abs(rotAngle) > 1e-3 and np.abs(rotAngle - np.pi) > 1e-3:
                    rotSin = np.sqrt(1 - rotCos * rotCos )
                    rotAxis_x = 0.5 / rotSin * (rotMat[2, 1] - rotMat[1, 2] )
                    rotAxis_y = 0.5 / rotSin * (rotMat[0, 2] - rotMat[2, 0] )
                    rotAxis_z = 0.5 / rotSin * (rotMat[1, 0] - rotMat[0, 1] )

                    norm = rotAxis_x * rotAxis_x \
                            + rotAxis_y * rotAxis_y \
                            + rotAxis_z * rotAxis_z
                    norm = np.sqrt(norm )

                    rotate = et.SubElement(transform, 'rotate')
                    rotate.set('x', '%.6f' % (rotAxis_x / norm ) )
                    rotate.set('y', '%.6f' % (rotAxis_y / norm ) )
                    rotate.set('z', '%.6f' % (rotAxis_z / norm ) )
                    rotate.set('angle', '%.6f' % (rotAngle / np.pi * 180 ) )

            elif tr[0] == 't':
                t = tr[1]
                trans = et.SubElement(transform, 'translate')
                trans.set('x', '%.6f' % t[0] )
                trans.set('y', '%.6f' % t[1] )
                trans.set('z', '%.6f' % t[2] )
            else:
                print('Wrong: unrecognizable type of transformation!' )
                assert(False )

    if materials is not None:
        for mat in materials:
            matName, partId = mat[1], mat[0]
            bsdf = et.SubElement(shape, 'ref' )
            bsdf.set('name', 'bsdf')
            bsdf.set('id', name + '_' + partId )
    return root


def addMaterial(root, name, materials, adobeRootAbs, uvScaleValue = None ):
    for mat in materials:
        matName, partId = mat[1], mat[0]
        bsdf = et.SubElement(root, 'bsdf' )
        bsdf.set('type', 'microfacet')
        bsdf.set('id', name + '_' + partId )
        matId = matName.split('/')[-1]
        matFile = osp.join(adobeRootAbs, matId, 'mat.txt')

        if not osp.isfile(matFile ):
            bsdf.set('type', 'microfacet')
            # Add uv scale
            if uvScaleValue is not None:
                uvScale = et.SubElement(bsdf, 'float')
                uvScale.set('name', 'uvScale')
                uvScale.set('value', '%.3f' % uvScaleValue )

            # Add new albedo
            albedo = et.SubElement(bsdf, 'texture' )
            albedo.set('name', 'albedo' )
            albedo.set('type', 'bitmap' )
            albedofile = et.SubElement(albedo, 'string' )
            albedofile.set('name', 'filename' )
            albedofile.set('value', osp.join(matName, 'tiled', 'diffuse_tiled.png') )

            # Add albedo scale
            albedoScale = et.SubElement(bsdf, 'rgb')
            albedoScale.set('name', 'albedoScale')
            albedoScaleValue = np.random.random(3) * 0.6 + 0.7
            albedoScale.set('value', '%.3f %.3f %.3f' %
                    (albedoScaleValue[0], albedoScaleValue[1], albedoScaleValue[2] ) )

            # Add new normal
            normal = et.SubElement(bsdf, 'texture' )
            normal.set('name', 'normal')
            normal.set('type', 'bitmap')
            normalfile = et.SubElement(normal, 'string')
            normalfile.set('name', 'filename')
            normalfile.set('value', osp.join(matName, 'tiled', 'normal_tiled.png') )

            # Add new roughness
            roughness = et.SubElement(bsdf, 'texture' )
            roughness.set('name', 'roughness')
            roughness.set('type', 'bitmap')
            roughnessfile = et.SubElement(roughness, 'string')
            roughnessfile.set('name', 'filename')
            roughnessfile.set('value', osp.join(matName, 'tiled', 'rough_tiled.png') )

            # Add roughness scale
            roughScale = et.SubElement(bsdf, 'float')
            roughScale.set('name', 'roughnessScale')
            roughScaleValue = np.random.random() * 0.4 + 0.8
            roughScale.set('value', '%.3f' % roughScaleValue  )

        else:
            with open(matFile, 'r') as matIn:
                lines = matIn.readlines()
                lines = [x.strip() for x in lines if len(x.strip() ) > 0]
                albedoStr = lines[0]
                albedoStr = albedoStr.split(' ')
                albedoValue = []
                for n in range(0, 3):
                    albedoValue.append(float(albedoStr[n] ) )
                roughValue = float(lines[1] )
            albedo = et.SubElement(bsdf, 'rgb')
            albedo.set('name', 'albedo')
            albedo.set('value', '%.3f %.3f %.3f' % (albedoValue[0], albedoValue[1], albedoValue[2] ) )

            # Add albedo scale
            albedoScale = et.SubElement(bsdf, 'rgb')
            albedoScale.set('name', 'albedoScale')
            albedoScaleValue = np.random.random(3) * 0.6 + 0.7
            albedoScale.set('value', '%.3f %.3f %.3f' %
                    (albedoScaleValue[0], albedoScaleValue[1], albedoScaleValue[2] ) )


            rough = et.SubElement(bsdf, 'float')
            rough.set('name', 'roughness')
            rough.set('value', '%.3f' % roughValue )

            # Add roughness scale
            roughScale = et.SubElement(bsdf, 'float')
            roughScale.set('name', 'roughnessScale')
            roughScaleValue = np.random.random() * 1.0 + 0.5
            roughScale.set('value', '%.3f' % roughScaleValue  )

    return root


def addEnvmap(root, envmapName, envRoot, mean, std):
    envmapName = osp.join(envRoot, envmapName )
    scaleValue = max(np.random.randn() * std + mean, 20)

    emitter = et.SubElement(root, 'emitter')
    emitter.set('type', 'envmap')
    filename = et.SubElement(emitter, 'string')
    filename.set('name', 'filename')
    filename.set('value', envmapName )
    scale = et.SubElement(emitter, 'float')
    scale.set('name', 'scale')
    scale.set('value', '%.4f' % (scaleValue ) )

    return root


def sampleRadianceFromTemp( lowTemp = 4000, highTemp = 8000 ):
    tempRadPair = {}
    tempRadPair[4000] = np.array([4892, 3202, 1846], dtype=np.float32 )
    tempRadPair[5000] = np.array([15248, 12072, 9608], dtype=np.float32 )
    tempRadPair[6000] = np.array([32704, 29344, 28608], dtype=np.float32 )
    tempRadPair[7000] = np.array([56672, 55648, 62496], dtype=np.float32 )
    tempRadPair[8000] = np.array([86349, 90277, 112622], dtype=np.float32 )

    sampledTemp = lowTemp + np.random.random() * (highTemp - lowTemp )
    sd = int(int(sampledTemp / 1000 ) * 1000 )
    sd = max(sd, lowTemp )
    su = min(sd + 1000, highTemp )
    wd = (su - sampledTemp) / 1000.0
    wu = 1 - wd
    rgb = tempRadPair[sd] * wd + tempRadPair[su] * wu

    return rgb / 120

def addAreaLight(root, name, fileName, transforms = None ):
    shape = et.SubElement(root, 'shape')
    shape.set('id', '%s_object' % name )
    shape.set('type', 'obj' )

    stringF = et.SubElement(shape, 'string' )
    stringF.set('name', 'filename' )
    stringF.set('value', fileName )

    emitter = et.SubElement(shape, 'emitter')
    emitter.set('type', 'area')


    rgbColor = sampleRadianceFromTemp()
    rgb = et.SubElement(emitter, 'rgb')
    rgb.set('value', '%.3f %.3f %.3f' % (rgbColor[0], rgbColor[1], rgbColor[2] ) )

    if transforms != None:
        transform = et.SubElement(shape, 'transform')
        transform.set('name', 'toWorld')
        for tr in transforms:
            if tr[0] == 's':
                s = tr[1]
                scale = et.SubElement(transform, 'scale' )
                scale.set('x', '%.6f' % s[0] )
                scale.set('y', '%.6f' % s[1] )
                scale.set('z', '%.6f' % s[2] )

            elif tr[0] == 'rot':
                rotMat = tr[1]
                rotTr = rotMat[0,0] + rotMat[1,1] + rotMat[2,2]
                rotCos = (rotTr - 1) * 0.5
                rotAngle = np.arccos(np.clip(rotCos, -1, 1 ) )
                if np.abs(rotAngle) > 1e-2:
                    rotSin = np.sqrt(1 - rotCos * rotCos )
                    rotAxis_x = 0.5 / rotSin * (rotMat[2, 1] - rotMat[1, 2] )
                    rotAxis_y = 0.5 / rotSin * (rotMat[0, 2] - rotMat[2, 0] )
                    rotAxis_z = 0.5 / rotSin * (rotMat[1, 0] - rotMat[0, 1] )

                    norm = rotAxis_x * rotAxis_x \
                            + rotAxis_y * rotAxis_y \
                            + rotAxis_z * rotAxis_z
                    norm = np.sqrt(norm )

                    rotate = et.SubElement(transform, 'rotate' )
                    rotate.set('x', '%.6f' % (rotAxis_x / norm ) )
                    rotate.set('y', '%.6f' % (rotAxis_y / norm ) )
                    rotate.set('z', '%.6f' % (rotAxis_z / norm ) )
                    rotate.set('angle', '%.6f' % (rotAngle / np.pi * 180 ) )

            elif tr[0] == 't':
                t = tr[1]
                trans = et.SubElement(transform, 'translate')
                trans.set('x', '%.6f' % t[0] )
                trans.set('y', '%.6f' % t[1] )
                trans.set('z', '%.6f' % t[2] )
            else:
                print('Wrong: unrecognizable type of transformation!' )
                assert(False )
    return root


def addSensor(root, imWidth, imHeight, intMat, sampleCount = 64 ):
    tanValue = float(1296 ) / 2.0 / float(intMat[0, 0] )
    angle = np.arctan(tanValue ) * 2
    angle = angle / np.pi * 180.0

    camera = et.SubElement(root, 'sensor')
    camera.set('type', 'perspective')
    fov = et.SubElement(camera, 'float')
    fov.set('name', 'fov')
    fov.set('value', '%.4f' % (angle ) )
    fovAxis = et.SubElement(camera, 'string')
    fovAxis.set('name', 'fovAxis')
    fovAxis.set('value', 'x')
    film = et.SubElement(camera, 'film')
    film.set('type', 'hdrfilm')
    width = et.SubElement(film, 'integer')
    width.set('name', 'width')
    width.set('value', '%d' % (imWidth) )
    height = et.SubElement(film, 'integer')
    height.set('name', 'height')
    height.set('value', '%d' % (imHeight) )
    sampler = et.SubElement(camera, 'sampler')
    sampler.set('type', 'adaptive')
    sampleNum = et.SubElement(sampler, 'integer')
    sampleNum.set('name', 'sampleCount')
    sampleNum.set('value', '%d' % (sampleCount) )

    return root


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


def readBox(boxName ):
    with open(boxName, 'r') as fIn:
        lines = fIn.readlines()
    lines = [x.strip() for x in lines if len(x.strip()) > 0 ]
    corners = []
    for l in lines:
        lArr = [float(x) for x in l.split(' ') if len(x.strip() ) > 0 ]
        corners.append(np.array(lArr ).reshape(1, 3) )
    corners = np.concatenate(corners, axis=0 )
    corners = np.loadtxt(boxName )
    return corners


def readMatObjectAnn(annName, annNumName ):
    if not osp.isfile(annName ) or not osp.isfile(annNumName ):
        return False, None

    with open(annNumName, 'r') as fIn:
        matNum = int(fIn.readline().strip() )

    with open(annName, 'r') as fIn:
        lines = fIn.readlines()
    lines = [x.strip() for x in lines if len(x.strip() ) != 0 ]
    if lines[0][0] == '!':
        return False, None

    matList = {}
    for l in lines:
        if l[0] == '#':
            continue
        else:
            lArr = [int(x) for x in l.split(' ') if len(x.strip()) != 0]
            partId = lArr[0]
            mats = lArr[1:]
            if partId < matNum:
                if not 9 in mats:
                    mats = [matCatList[x] for x in mats ]
                    matList[partId ] = mats
    return True, matList


def readSceneList(sceneListDir ):
    wallListName = osp.join(sceneListDir, 'wall.txt' )
    floorListName = osp.join(sceneListDir, 'floor.txt' )
    sceneListName = osp.join(sceneListDir, 'scene_type.txt' )

    wallMatPair = {}
    with open(wallListName, 'r') as fIn:
        lines = fIn.readlines()
    for l in lines:
        l = l.strip()
        sceneType, matTypes = l.split(':')[0], l.split(':')[1]
        matTypes = [x.strip() for x in matTypes.split(',')
                if len(x.strip()) > 0 ]
        wallMatPair[sceneType.lower().replace(' ', '') ] = matTypes

    floorMatPair = {}
    with open(floorListName, 'r') as fIn:
        lines = fIn.readlines()
    for l in lines:
        l = l.strip()
        sceneType, matTypes = l.split(':')[0], l.split(':')[1]
        matTypes = [x.strip() for x in matTypes.split(',')
                if len(x.strip()) > 0 ]
        floorMatPair[sceneType.lower().replace(' ', '') ] = matTypes

    sceneWallMatPair = {}
    sceneFloorMatPair = {}
    with open(sceneListName, 'r') as fIn:
        lines = fIn.readlines()
    for l in lines:
        l = l.strip()
        sceneId = l[0:12]
        sceneTypes = l[12:].split('/')
        sceneTypes = [x.strip() for x in sceneTypes ]

        matWallTypes = []
        matFloorTypes = []
        for sceneType in sceneTypes:
            mwts = wallMatPair[sceneType.lower().replace(' ', '') ]
            for t in mwts:
                if not t in matWallTypes:
                    matWallTypes.append(t )

            mfts = floorMatPair[sceneType.lower().replace(' ', '') ]
            for t in mfts:
                if not t in matFloorTypes:
                    matFloorTypes.append(t )

        sceneWallMatPair[sceneId ] = matWallTypes
        sceneFloorMatPair[sceneId ] = matFloorTypes

    return sceneWallMatPair, sceneFloorMatPair


def readMatList(matRoot ):
    # Load objList
    objList = {}
    for matType in matCatList:
        fileName = osp.join(matRoot, matType + '.txt')
        with open(fileName, 'r') as fIn:
            lines = fIn.readlines()
        lines = [l.strip() for l in lines if len(l.strip() ) > 0]
        objList[matType ] = lines

    # Load wallList
    wallList = {}
    wallFiles = glob.glob(osp.join(matRoot, '*_wall.txt') )
    for wallFile in wallFiles:
        matType = wallFile.split('/')[-1].replace('_wall.txt', '')
        with open(wallFile, 'r') as fIn:
            lines = fIn.readlines()
        lines = [l.strip() for l in lines if len(l.strip() ) > 0 ]
        wallList[matType ] = lines

    # Load floorList
    floorList = {}
    floorFiles = glob.glob(osp.join(matRoot, '*_floor.txt') )
    for floorFile in floorFiles:
        matType = floorFile.split('/')[-1].replace('_floor.txt', '')
        with open(floorFile, 'r') as fIn:
            lines = fIn.readlines()
        lines = [l.strip() for l in lines if len(l.strip() ) > 0 ]
        floorList[matType ] = lines

    return objList, wallList, floorList


def readEnvList(envRoot ):
    envList = glob.glob(osp.join(envRoot, '*.hdr') )
    envList = [x.split('/')[-1] for x in envList ]
    return envList



def changeToNewLight(root, mean, std, isWindow ):
    ## Change the emitter
    if isWindow:
        isArea = np.random.random() > 0.5
    else:
        isArea = True

    if isArea == True:
        isEnv = np.random.random() > 0.5
    else:
        isEnv = True

    if isArea == False:
        print('Warning: the indoor light will be turned off.')
    if isEnv == False:
        print('Warning: the envmap will be turned dark.')

    if isArea:
        for shape in root.iter('shape'):
            emitters = shape.findall('emitter')
            for emitter in emitters:
                eType = emitter.get('type')
                if eType == 'area':
                    rgb = emitter.findall('rgb')[0]
                    rgbColor = sampleRadianceFromTemp()
                    rgb.set('value', '%.3f %.3f %.3f' % (rgbColor[0], rgbColor[1], rgbColor[2] ) )
    else:
        for shape in root.iter('shape'):
            emitters = shape.findall('emitter')
            for emitter in emitters:
                shape.remove(emitter )

    for emitter in root.iter('emitter' ):
        eType = emitter.get('type')
        if eType == 'envmap':
            floats = emitter.findall('float')[0]
            assert(floats.get('name') == 'scale')
            if isEnv == True:
                scale = max(np.random.randn() * std + mean, 20)
                floats.set('value', '%.4f' % scale )
            else:
                floats.set('value', '0.00000001')
    return root




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Directories
    parser.add_argument('--out', default="../scenes/xml1/", help="outdir of xml file" )
    parser.add_argument('--annotation', default='/siggraphasia20dataset/full_annotations.json', help='the file of the annotation' )
    # Rendering parameters
    parser.add_argument('--width', default=640, type=int, help='the width of the image' )
    parser.add_argument('--height', default=480, type=int, help='the height of the image' )
    parser.add_argument('--sampleCount', default=1024, type=int, help='the by default number of samples' )
    # Material lists
    parser.add_argument('--matList', default='./MatLists/', help='the list of materials for objects' )
    parser.add_argument('--sceneList', default='./forcedFloorTemp.txt', help='the list of scene to be processed')
    parser.add_argument('--sceneMatList', default='./MatSceneLists/', help='the list of materials for scenes' )
    # Lighting parameters
    parser.add_argument('--envScaleMean', default=120, type=float, help='the mean of envScale' )
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
    adobeRootAbs = params['adobestockAbs']
    camRootAbs = params['scannet_camAbs']
    envRootAbs = params['envmapAbs']

    doorDirs = glob.glob(osp.join(shapeNetRootAbs, 'door', '*') )
    doorDirs = [x for x in doorDirs if osp.isdir(x) ]
    windowDirs = glob.glob(osp.join(shapeNetRootAbs, 'window', '*') )
    windowDirs = [x for x in windowDirs if osp.isdir(x) ]
    curtainDirs = glob.glob(osp.join(shapeNetRootAbs, 'curtain', '*') )
    curtainDirs = [x for x in curtainDirs if osp.isdir(x) ]
    cLightDirs = glob.glob(osp.join(shapeNetRootAbs, 'ceiling_lamp', '*') )
    cLightDirs = [x for x in cLightDirs if osp.isdir(x) ]

    sceneWallMatPair, sceneFloorMatPair = readSceneList(opt.sceneMatList )
    objList, wallList, floorList = readMatList(opt.matList )
    envList = readEnvList(envRootAbs )

    if not opt.sceneList is None:
        with open(opt.sceneList, 'r') as fIn:
            sceneIdList = fIn.readlines()
        sceneIdList = [x.strip() for x in sceneIdList ]

    sceneCnt = 0
    for r in JSONHelper.read(filename_json ):
        if not(sceneCnt >= opt.rs and sceneCnt < opt.re):
            continue
        sceneCnt += 1

        id_scan = r["id_scan"]
        if not opt.sceneList is None:
            if not id_scan in sceneIdList:
                continue

        outdir = osp.join(opt.out, id_scan)
        xmlOutFile = osp.join(outdir, 'main.xml')
        transformFile = osp.join(outdir, 'transform.dat').replace('/xml/', '/xml1/')
        if not osp.isfile(transformFile ):
            continue

        print('%d/%d: %s' % (sceneCnt, opt.re, id_scan ) )
        os.system('mkdir -p %s' % outdir )

        # load transformations
        with open(transformFile, 'rb') as fIn:
            transforms = pickle.load(fIn )

        root  = et.Element('scene')
        root.set('verion', '0.5.0')
        integrator = et.SubElement(root, 'integrator')
        integrator.set('type', 'path')

        ########################################################################################
        # Write layout to the xml file
        layOutMesh = osp.join(layoutRoot, id_scan, 'uv_mapped.obj' )
        containerMesh = osp.join(layoutRoot, id_scan, 'container.obj' )

        wallMatTypes = sceneWallMatPair[id_scan ]
        floorMatTypes = sceneFloorMatPair[id_scan ]
        wallMatType = wallMatTypes[np.random.randint(len(wallMatTypes ) ) ]
        floorMatType = floorMatTypes[np.random.randint(len(floorMatTypes ) ) ]

        wallMats = wallList[wallMatType ]
        floorMats = floorList[floorMatType ]
        wallMat = wallMats[np.random.randint(len(wallMats ) ) ]
        floorMat = floorMats[np.random.randint(len(floorMats ) ) ]
        wallMat = osp.join(adobeRoot, wallMat )
        floorMat = osp.join(adobeRoot, floorMat )
        materials = [('wall', wallMat), ('ceiling', wallMat), ('floor', floorMat) ]

        uvScale = 0.2 + 0.2 * np.random.random()
        root = addMaterial(root, id_scan, materials, adobeRootAbs, uvScale )
        root = addShape(root, id_scan, layOutMesh, transforms[0], materials )
        root = addShape(root, id_scan, containerMesh, transforms[0], materials )

        ########################################################################################
        # Write shapes to the xml file
        shapeCnt = 1
        modelMatPair = {}
        for model in r["aligned_models"]:
            id_cad = model["id_cad"]
            catid_cad = model["catid_cad"]

            annFileName = osp.join(shapeNetRootAbs, catid_cad, id_cad, 'ann.txt')
            annNumFileName = osp.join(shapeNetRootAbs, catid_cad, id_cad, 'matNum.txt' )
            isValid, anns = readMatObjectAnn(annFileName, annNumFileName )
            if isValid == False:
                print('Warning %s/%s is a low quality model.' % (catid_cad, id_cad ) )
                continue

            if not id_cad in modelMatPair:
                materials = []
                for partId, matTypes in anns.items():
                    matType = matTypes[np.random.randint(len(matTypes ) ) ]

                    objMats = objList[matType ]
                    objMat = objMats[np.random.randint(len(objMats ) ) ]
                    objMat = osp.join(adobeRoot, objMat )
                    materials.append( ('part%d' % partId, objMat ) )

                modelMatPair[id_cad ] = materials
                uvScale = 0.7 + 0.6 * np.random.random()
                root = addMaterial(root, catid_cad + '_' + id_cad, materials, adobeRootAbs, uvScale )
            else:
                materials = modelMatPair[id_cad ]


            if catid_cad != '03636649':
                cad_file = osp.join(shapeNetRoot, catid_cad, id_cad, 'alignedNew.obj' )
                cad_fileAbs = osp.join(shapeNetRootAbs, catid_cad, id_cad, 'alignedNew.obj')
                if not osp.isfile(cad_fileAbs ):
                    continue
                root = addShape(root, catid_cad + '_' + id_cad, cad_file, transforms[shapeCnt ], materials )
            else:
                cad_fileAbs = osp.join(shapeNetRootAbs, catid_cad, id_cad, 'alignedNew.obj')
                if not osp.isfile(cad_fileAbs ):
                    continue
                cad_fileAbs = osp.join(shapeNetRootAbs, catid_cad, id_cad, 'aligned_light.obj')
                if osp.isfile(cad_fileAbs ):
                    cad_file = osp.join(shapeNetRoot, catid_cad, id_cad, 'aligned_shape.obj' )
                    root = addShape(root, catid_cad + '_' + id_cad, cad_file, transforms[shapeCnt ], materials )
                    cad_file = osp.join(shapeNetRoot, catid_cad, id_cad, 'aligned_light.obj' )
                    root = addAreaLight(root, catid_cad + '_' + id_cad, cad_file, transforms[shapeCnt ] )
                else:
                    cad_file = osp.join(shapeNetRoot, catid_cad, id_cad, 'alignedNew.obj' )
                    root = addAreaLight(root, catid_cad + '_' + id_cad, cad_file, transforms[shapeCnt ] )
            shapeCnt += 1

        ##########################################################################################
        # Write door, window curtain to the xml file
        isWindow = False

        random.shuffle(doorDirs )
        random.shuffle(windowDirs )
        random.shuffle(curtainDirs )
        random.shuffle(cLightDirs )
        doorDir = osp.join(shapeNetRoot, '/'.join(doorDirs[0].split('/')[-2:] ) )
        windowDir = osp.join(shapeNetRoot, '/'.join(windowDirs[0].split('/')[-2:] ) )
        curtainDir = osp.join(shapeNetRoot, '/'.join(curtainDirs[0].split('/')[-2:] ) )
        cLightDir = osp.join(shapeNetRoot, '/'.join(cLightDirs[0].split('/')[-2:] ) )

        doorDirAbs = doorDirs[0]
        doorMaterials = []
        annFileName = osp.join(doorDirAbs, 'ann.txt' )
        annNumFileName = osp.join(doorDirAbs, 'matNum.txt' )
        isValid, anns = readMatObjectAnn(annFileName, annNumFileName )
        if isValid == False:
            print(annFileName, annNumFileName, anns )
            assert(isValid == True )
        for partId, matTypes in anns.items():
            matType = matTypes[np.random.randint(len(matTypes ) ) ]
            objMats = objList[matType ]
            objMat = objMats[np.random.randint(len(objMats ) ) ]
            objMat = osp.join(adobeRoot, objMat )
            doorMaterials.append( ('part%d' % partId, objMat ) )
        uvScale = 0.7 + 0.6 * np.random.random()
        root = addMaterial(root, 'door_' + doorDir.split('/')[-1], doorMaterials, adobeRootAbs, uvScale )

        windowDirAbs = windowDirs[0]
        windowMaterials = []
        annFileName = osp.join(windowDirAbs, 'ann.txt' )
        annNumFileName = osp.join(windowDirAbs, 'matNum.txt' )
        isValid, anns = readMatObjectAnn(annFileName, annNumFileName )
        assert(isValid == True )
        for partId, matTypes in anns.items():
            matType = matTypes[np.random.randint(len(matTypes ) ) ]
            objMats = objList[matType ]
            objMat = objMats[np.random.randint(len(objMats ) ) ]
            objMat = osp.join(adobeRoot, objMat )
            windowMaterials.append( ('part%d' % partId, objMat ) )
        uvScale = 0.7 + 0.6 * np.random.random()
        root = addMaterial(root, 'window_' + windowDir.split('/')[-1], windowMaterials, adobeRootAbs, uvScale )

        curtainDirAbs = curtainDirs[0]
        curtainMaterials = []
        annFileName = osp.join(curtainDirAbs, 'ann.txt' )
        annNumFileName = osp.join(curtainDirAbs, 'matNum.txt' )
        isValid, anns = readMatObjectAnn(annFileName, annNumFileName )
        assert(isValid == True )
        for partId, matTypes in anns.items():
            matType = matTypes[np.random.randint(len(matTypes ) ) ]
            objMats = objList[matType ]
            objMat = objMats[np.random.randint(len(objMats ) ) ]
            objMat = osp.join(adobeRoot, objMat )
            curtainMaterials.append( ('part%d' % partId, objMat ) )
        uvScale = 0.7 + 0.6 * np.random.random()
        root = addMaterial(root, 'curtain_' + curtainDir.split('/')[-1], curtainMaterials, adobeRootAbs, uvScale )

        cLightDirAbs = cLightDirs[0]
        cLightMaterials = []
        annFileName = osp.join(cLightDirAbs, 'ann.txt')
        annNumFileName = osp.join(cLightDirAbs, 'matNum.txt')
        isValid, anns = readMatObjectAnn(annFileName, annNumFileName )
        assert(isValid == True )
        for partId, matTypes in anns.items():
            matType = matTypes[np.random.randint(len(matTypes ) ) ]
            objMats = objList[matType ]
            objMat = objMats[np.random.randint(len(objMats ) ) ]
            objMat = osp.join(adobeRoot, objMat )
            cLightMaterials.append( ('part%d' % partId, objMat ) )
        uvScale = 0.7 + 0.6 * np.random.random()
        root = addMaterial(root, 'ceiling_lamp_' + cLightDir.split('/')[-1], cLightMaterials, adobeRootAbs, uvScale )

        with open(osp.join(outdir, 'dwcl_config.txt'), 'w') as fOut:
            fOut.write('d: %s\n' % doorDir.split('/')[-1] )
            fOut.write('w: %s\n' % windowDir.split('/')[-1] )
            fOut.write('c: %s\n' % curtainDir.split('/')[-1] )
            fOut.write('l: %s\n' % cLightDir.split('/')[-1] )

        doorBoxName = osp.join(doorDirAbs, 'bbox.txt')
        doorBox = readBox(doorBoxName )
        windowBoxName = osp.join(windowDirAbs, 'bbox.txt')
        windowBox = readBox(windowBoxName )
        curtainBoxName = osp.join(curtainDirAbs, 'bbox.txt')
        curtainBox = readBox(curtainBoxName )
        cLightBoxName = osp.join(cLightDirAbs, 'bbox.txt')
        cLightBox = readBox(cLightBoxName )

        isCurtain = (np.random.random() > 0.5 )
        layoutDir = osp.join(layoutRootAbs, id_scan )
        cornerFile = osp.join(layoutDir, id_scan + '_corners.npy' )
        corners = np.load(cornerFile, allow_pickle=True ).item()

        # Add light
        cLight_corners = utils.get_light_corners(corners['light_ctr'], cLightBox )
        trCLight = utils.get_transform(cLight_corners, cLightBox, 'li' )
        trCLight = trCLight + transforms[0]

        cad_fileAbs = osp.join(cLightDirAbs, 'aligned_light.obj')
        if osp.isfile(cad_fileAbs ):
            cad_file = osp.join(cLightDir, 'aligned_shape.obj' )
            root = addShape(root, 'ceiling_lamp_' + cLightDir.split('/')[-1], cad_file, trCLight, cLightMaterials )
            cad_file = osp.join(cLightDir, 'aligned_light.obj' )
            root = addAreaLight(root, 'ceiling_lamp_' + cLightDir.split('/')[-1], cad_file, trCLight )
        else:
            cad_file = osp.join(cLightDir, 'alignedNew.obj' )
            root = addAreaLight(root,  'ceiling_lamp_' + cLightDir.split('/')[-1], cad_file, trCLight )

        # Add door window and curtain
        objTypes = corners['type']
        coords = corners['coords']
        assert(len(objTypes ) == len(coords ) )
        for n in range(0, len(objTypes ) ):
            objType = objTypes[n]
            coord = coords[n]
            if objType == 'w':
                isWindow = True
                trWindow  = utils.get_transform(coord, windowBox, 'w')
                trWindow = trWindow + transforms[0]
                cad_file = osp.join(windowDir, 'alignedNew.obj')
                root = addShape(root, 'window_' + windowDir.split('/')[-1], cad_file,
                        trWindow, windowMaterials )
                if isCurtain:
                    trCurtain = utils.get_transform(coord, curtainBox, 'c' )
                    trCurtain = trCurtain + transforms[0]
                    cad_file = osp.join(curtainDir, 'alignedNew.obj' )
                    root = addShape(root, 'curtain_' + curtainDir.split('/')[-1], cad_file,
                            trCurtain, curtainMaterials )
            elif objType == 'd':
                trDoor = utils.get_transform(coord, doorBox, 'd')
                trDoor = trDoor + transforms[0]
                cad_file =osp.join(doorDir, 'alignedNew.obj')
                root = addShape(root, 'door_' + doorDir.split('/')[-1], cad_file,
                        trDoor, doorMaterials )
            else:
                print('Unrecognizable object type')
                assert(False )

        #############################################################################################
        # Write camera intrinsic to xml file
        intDir = osp.join(osp.join(camRootAbs, id_scan), 'intrinsic')
        intFile = osp.join(intDir, 'intrinsic_color.txt')

        intMat = np.zeros((3, 3), dtype=np.float32 )
        with open(intFile, 'r') as intIn:
            for n in range(0, 3):
                intLine = intIn.readline().strip()
                intLine = [float(x) for x in intLine.split(' ')]
                for m in range(0, 3):
                    intMat[n, m] = intLine[m]
        root = addSensor(root, opt.width, opt.height, intMat, sampleCount = opt.sampleCount )

        ############################################################################################
        # Write environment map to xml file
        envId = envList[np.random.randint(len(envList ) ) ]
        root = addEnvmap(root, envId, envRoot, opt.envScaleMean, opt.envScaleStd )


        ###########################################
        # change to new light
        root = changeToNewLight(root, opt.envScaleMean, opt.envScaleStd, isWindow )

        ############################################################################################
        # Create xml file
        xmlString = transformToXml(root )
        with open(xmlOutFile, 'w') as xmlOut:
            xmlOut.write(xmlString )
