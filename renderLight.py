import glob
import os
import os.path as osp
import argparse
import numpy as np

parser = argparse.ArgumentParser()
# Directories
parser.add_argument('--xmlRoot', default="/siggraphasia20dataset/code/Routine/scenes/xml1", help="outdir of xml file")
# Start and end point
parser.add_argument('--rs', default=3, type=int, help='the width of the image' )
parser.add_argument('--re', default=4, type=int, help='the height of the image' )
# xml file
parser.add_argument('--xmlFile', default='main', help='the xml file')
# output file
parser.add_argument('--outRoot', default='/eccv20dataset/DatasetNew_test', help='output directory')
# Control
parser.add_argument('--forceOutput', action='store_true', help='whether to overwrite previous results')
# Program
parser.add_argument('--program', default='/siggraphasia20dataset/OptixRendererLight/src/bin/optixRenderer', help='the location of render' )
opt = parser.parse_args()

scenes = glob.glob(osp.join(opt.xmlRoot, 'scene*') )
scenes = [x for x in scenes if osp.isdir(x) ]
scenes = sorted(scenes )
for n in range(opt.rs, min(opt.re, len(scenes ) ) ):
    scene = scenes[n]
    sceneId = scene.split('/')[-1]

    print('%d/%d: %s' % (n, len(scenes), sceneId ) )

    outDir = osp.join(opt.outRoot, opt.xmlFile + '_' + opt.xmlRoot.split('/')[-1], sceneId )
    os.system('mkdir -p %s' % outDir )

    xmlFile = osp.join(scene, '%s.xml' % opt.xmlFile )
    camFile = osp.join(scene, 'cam.txt' )
    if not osp.isfile(xmlFile ) or not osp.isfile(camFile ):
        continue

    imNames = glob.glob(osp.join(outDir, 'im_*.hdr') )
    envNames = glob.glob(osp.join(outDir, 'imenv_*.hdr') )
    #if len(envNames ) == len(imNames ):
    #    continue

    cmd = '%s -f %s -c %s -o %s -m %d' % (opt.program, xmlFile, 'cam.txt', osp.join(outDir, 'im.rgbe'), 7 )


    if opt.forceOutput:
        cmd += ' --forceOutput'

    os.system(cmd )
