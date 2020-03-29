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
            roughScaleValue = np.random.random() * 1.0 + 0.5
            roughScale.set('value', '%.3f' % roughScaleValue  )

        else:
            with open(matFile, 'r') as matIn:
                lines = matIn.readlines()
                lines = [x.strip() for x in lines if len(x.strip() ) > 0]
                albedoStr = lines[0]
                albedoStr = albedoStr.split(' ')
                albedoValue = []
                for n in range(0, 3):
                    albedoValue.append(float(albedoStr[0] ) )
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


def sampleRadianceFromTemp(lowTemp = 4000, highTemp = 8000 ):
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
    return rgb / 200.0

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


def changeToNewLight(root, mean, std, isWindow, isArea ):
    ## Change the emitter
    if not isWindow:
        isArea = True

    if isArea == True:
        isEnv = np.random.random() > 0.15
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
    parser.add_argument('--out', default="./xml1/", help="outdir of xml file" )
    parser.add_argument('--annotation', default='/newfoundland/zhl/Scan2cad/full_annotations.json', help='the file of the annotation' )
    # Material lists
    parser.add_argument('--matList', default='./MatLists/', help='the list of materials for objects' )
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

    envRoot = params['envmap']
    envRootAbs = params['envmapAbs']
    shapeNetRoot = params['shapenet']
    shapeNetRootAbs = params['shapenetAbs']
    adobeRoot = params['adobestock']
    adobeRootAbs = params['adobestockAbs']
    layoutRoot = params["scannet_layout"]
    layoutRootAbs = params["scannet_layoutAbs"]

    cLightDirs = glob.glob(osp.join(shapeNetRootAbs, 'ceiling_lamp', '*') )
    cLightDirs = [x for x in cLightDirs if osp.isdir(x) ]

    objList, _, _ = readMatList(opt.matList )
    envList = readEnvList(envRootAbs )

    sceneCnt = 0
    for r in JSONHelper.read(filename_json ):
        if not(sceneCnt >= opt.rs and sceneCnt < opt.re):
            continue
        sceneCnt += 1

        id_scan = r["id_scan"]

        print('%d/%d: %s' % (sceneCnt, opt.re, id_scan ) )
        outdir = osp.join(opt.out, id_scan)

        #############################################################################################
        # load transformations
        transformFile = osp.join(outdir, 'transform.dat')
        if not osp.isfile(transformFile ):
            continue
        with open(transformFile, 'rb') as fIn:
            transforms = pickle.load(fIn)

        #############################################################################################
        # Delete old lighting
        oldXML = osp.join(outdir, 'main.xml' )
        newXML = osp.join(outdir, 'mainDiffLight.xml')

        tree  = et.parse(oldXML )
        root = tree.getroot()

        shapes = root.findall('shape')
        isArea = True
        for shape in shapes:
            emitters = shape.findall('emitter')
            if len(emitters ) > 0:
                isArea = False
                break

        envmapList = []
        emitters = root.findall('emitter')
        for emitter in emitters:
            if emitter.get('type') == 'envmap':
                envmapList.append(emitter )
        for emitter in envmapList:
            root.remove(emitter )

        ceilingList = []
        isWindow = False
        for shape in shapes:
            shapeId = shape.get('id')
            nameArr = shapeId.split('_')
            if nameArr[0] == 'window':
                isWindow = True
            elif nameArr[0] == 'ceiling':
                ceilingList.append(shape )
        for shape in ceilingList:
            root.remove(shape )

        ceilingMatList = []
        mats = root.findall('bsdf')
        for mat in mats:
            matId = mat.get('id')
            nameArr = matId.split('_')
            if nameArr[0] == 'ceiling':
                ceilingMatList.append(mat )
        for mat in ceilingMatList:
            root.remove(mat )

        ##########################################################################################
        # Write door, window curtain to the xml file
        random.shuffle(cLightDirs )
        cLightDir = osp.join(shapeNetRoot, '/'.join(cLightDirs[0].split('/')[-2:] ) )

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

        with open(osp.join(outdir, 'l_config.txt'), 'w') as fOut:
            fOut.write('l: %s\n' % cLightDir.split('/')[-1] )

        cLightBoxName = osp.join(cLightDirAbs, 'bbox.txt')
        cLightBox = readBox(cLightBoxName )

        layoutDir = osp.join(layoutRootAbs, id_scan )
        cornerFile = osp.join(layoutDir, id_scan + '_corners.npy' )
        corners = np.load(cornerFile ).item()

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

        ############################################################################################
        # Write environment map to xml file
        envId = envList[np.random.randint(len(envList ) ) ]
        root = addEnvmap(root, envId, envRoot, opt.envScaleMean, opt.envScaleStd )



        ###########################################
        # change to new light
        root = changeToNewLight(root, opt.envScaleMean, opt.envScaleStd, isWindow, isArea )

        ############################################################################################
        # Create xml file
        xmlString = transformToXml(root )
        with open(newXML, 'w') as xmlOut:
            xmlOut.write(xmlString )
