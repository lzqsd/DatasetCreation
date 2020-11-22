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
    # Directories
    parser.add_argument('--xmlRoot', default='/siggraphasia20dataset/code/Routine/scenes/xml1')
    parser.add_argument('--xmlFile', default="mainDiffLight")
    parser.add_argument('--outRoot', default='/siggraphasia20dataset/code/Routine/DatasetCreation/')
    parser.add_argument('--program', default='/siggraphasia20dataset/OptixRendererShading/src/bin/optixRenderer')
    # Start and end point
    parser.add_argument('--rs', default=13, type=int, help='the width of the image' )
    parser.add_argument('--re', default=14, type=int, help='the height of the image' )
    opt = parser.parse_args()

    xmlFile = opt.xmlFile
    xmlRoot = opt.xmlRoot

    outRoot = xmlFile.split('/')[-1] + '_' + xmlRoot.split('/')[-1]
    outRoot = osp.join(opt.outRoot, outRoot )
    scenes = glob.glob(osp.join(outRoot, 'scene*' ) )
    scenes = sorted(scenes )

    for k in range(opt.rs, min(opt.re, len(scenes ) ) ):
        outDir = scenes[k]
        if not osp.isdir(outDir ):
            continue

        sceneId = outDir.split('/')[-1]
        xmlDir = osp.join(xmlRoot, sceneId )

        xmlFile = osp.join(xmlDir, '%s.xml' % opt.xmlFile )
        if not osp.isfile(xmlFile ):
            continue

        print('%d/%d: %s' % (k, min(opt.re, len(scenes ) ), sceneId ) )

        newFile = osp.join(xmlDir, '%s-shading.xml' % opt.xmlFile )

        tree = et.parse(xmlFile )
        root  = tree.getroot()

        sensor = root.findall('sensor')[0]
        film = sensor.findall('film')[0]
        integers = film.findall('integer')
        for integer in integers:
            if integer.get('name') == 'height':
                integer.set('value', '120')
            elif integer.get('name') == 'width':
                integer.set('value', '160')

        xmlString = transformToXml(root )
        with open(newFile, 'w') as xmlOut:
            xmlOut.write(xmlString )


        cmd = '%s -c cam.txt -f %s -o %s -m 0 --maxIteration 7 --camStart 3 --camEnd 4 --forceOutput' \
                % (opt.program, newFile, osp.join(outDir, 'imshadingDirect.rgbe') )
        print(cmd )
        os.system(cmd )
