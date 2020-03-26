import glob
import os
import os.path as osp
import argparse


parser = argparse.ArgumentParser()
# Directories
parser.add_argument('--out', default="./xml/", help="outdir of xml file")
# Start and end point
parser.add_argument('--rs', default=0, type=int, help='the width of the image' )
parser.add_argument('--re', default=1600, type=int, help='the height of the image' )
# Render Mode
parser.add_argument('--mode', default=0, type=int, help='the information being rendered')
# Control
parser.add_argument('--forceOutput', action='store_true', help='whether to overwrite previous results')
parser.add_argument('--medianFilter', action='store_true', help='whether to use median filter')
# output dir
parser.add_argument('--outputDir', default='Images', help='the output dir')
# Program
parser.add_argument('--program', default='~/OptixRenderer/src/bin/optixRenderer', help='the location of render' )
opt = parser.parse_args()


scenes = glob.glob(osp.join(opt.out, 'scene*') )
scenes = [x for x in scenes if osp.isdir(x) ]
scenes = sorted(scenes )
for n in range(opt.rs, min(opt.re, len(scenes ) ) ):
    scene = scenes[n]
    sceneId = scene.split('/')[-1]

    if not sceneId == 'scene0582_00':
        continue

    print('%d/%d: %s' % (n, len(scenes), sceneId ) )

    outDir = osp.join(os.getcwd(), opt.outputDir, sceneId )
    os.system('mkdir -p %s' % outDir )

    xmlFile = osp.join(scene, 'main.xml' )
    camFile = osp.join(scene, 'cam.txt' )
    if not osp.isfile(xmlFile ) or not osp.isfile(camFile ):
        continue

    cmd = '%s -f %s -c %s -o %s -m %d' % (opt.program, xmlFile, 'cam.txt', osp.join(outDir, 'im.rgbe'), opt.mode )

    if opt.forceOutput:
        cmd += ' --forceOutput'

    if opt.medianFilter:
        cmd += ' --medianFilter'

    os.system(cmd )
