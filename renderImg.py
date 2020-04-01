import glob
import os
import os.path as osp
import argparse
import xml.etree.ElementTree as et


parser = argparse.ArgumentParser()
# Directories
parser.add_argument('--out', default="./xml", help="outdir of xml file")
# Start and end point
parser.add_argument('--rs', default=0, type=int, help='the width of the image' )
parser.add_argument('--re', default=1600, type=int, help='the height of the image' )
# xml file
parser.add_argument('--xml', default='main', help='the xml file')
# Render Mode
parser.add_argument('--mode', default=0, type=int, help='the information being rendered')
# Control
parser.add_argument('--forceOutput', action='store_true', help='whether to overwrite previous results')
parser.add_argument('--medianFilter', action='store_true', help='whether to use median filter')
# Program
parser.add_argument('--program', default='~/OptixRenderer/src/bin/optixRenderer', help='the location of render' )
opt = parser.parse_args()


scenes = glob.glob(osp.join(opt.out, 'scene*') )
scenes = [x for x in scenes if osp.isdir(x) ]
scenes = sorted(scenes )
for n in range(opt.rs, min(opt.re, len(scenes ) ) ):
    scene = scenes[n]
    sceneId = scene.split('/')[-1]

    print('%d/%d: %s' % (n, len(scenes), sceneId ) )

    outDir = osp.join(os.getcwd(), opt.xml + '_' + opt.out.split('/')[-1], sceneId )
    os.system('mkdir -p %s' % outDir )

    xmlFile = osp.join(scene, '%s.xml' % opt.xml )
    camFile = osp.join(scene, 'cam.txt' )
    if not osp.isfile(xmlFile ) or not osp.isfile(camFile ):
        continue

    tree  = et.parse(xmlFile )
    root = tree.getroot()

    shapes = root.findall('shape')
    isFindAreaLight = False
    for shape in shapes:
        emitters = shape.findall('emitter')
        if len(emitters ) > 0:
            isFindAreaLight = True
            break


    cmd = '%s -f %s -c %s -o %s -m %d' % (opt.program, xmlFile, 'cam.txt', osp.join(outDir, 'im.rgbe'), opt.mode )

    if opt.forceOutput:
        cmd += ' --forceOutput'

    if opt.medianFilter:
        cmd += ' --medianFilter'

    if not isFindAreaLight:
        print('Warning: no area light found, may need more samples.' )
        cmd += ' --maxIteration 2'

    os.system(cmd )
