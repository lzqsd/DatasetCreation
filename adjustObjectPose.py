import numpy as np
import os
import shutil
import glob
import JSONHelper
import quaternion
import argparse
import os.path as osp
import pickle
import align_utils as utils


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


def writeScene(name, boxes ):
    with open(name, 'w') as meshOut:
        vNum = 0
        for group in boxes:
            vertices = group[0]
            faces = group[1]
            for n in range(0, vertices.shape[0] ):
                meshOut.write('v %.3f %.3f %.3f\n' %
                        (vertices[n, 0], vertices[n, 1], vertices[n, 2] ) )
            for n in range(0, faces.shape[0]):
                meshOut.write('f %d %d %d\n' %
                        (faces[n, 0] + vNum, faces[n, 1] + vNum, faces[n, 2] + vNum ) )
            vNum += vertices.shape[0]


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


def computeTransform(vertices, t, q, s):
    if s != None:
        scale = np.array(s, dtype=np.float32 ).reshape(1, 3)
        vertices = vertices * scale

    if q != None:
        q = np.quaternion(q[0], q[1], q[2], q[3])
        rotMat = quaternion.as_rotation_matrix(q )
        if np.abs(rotMat[1, 1] ) > 0.5:
            d = rotMat[1, 1]
            rotMat[:, 1] = 0
            rotMat[1, :] = 0
            if d < 0:
                rotMat[1, 1] = -1
            else:
                rotMat[1, 1] = 1
        vertices = np.matmul(rotMat, vertices.transpose() )
        vertices = vertices.transpose()

    if t != None:
        trans = np.array(t, dtype=np.float32 ).reshape(1, 3)
        vertices = vertices + trans

    return vertices, trans.squeeze(), rotMat, scale.squeeze()


def checkOverlapApproximate(bverts1, bverts2 ):
    axis_1 = (bverts1[1, :] - bverts1[0, :] ).reshape(1, 3)
    xLen = np.sqrt(np.sum(axis_1 * axis_1 ) )
    axis_2 = (bverts1[3, :] - bverts1[0, :] ).reshape(1, 3)
    zLen = np.sqrt(np.sum(axis_2 * axis_2 ) )

    origin = bverts1[0, :]
    xCoord = np.sum( (bverts2[0:4, :] - origin ) * axis_1 / xLen, axis=1 )
    zCoord = np.sum( (bverts2[0:4, :] - origin ) * axis_2 / zLen, axis=1 )
    minX, maxX = xCoord.min(), xCoord.max()
    minZ, maxZ = zCoord.min(), zCoord.max()

    xOverlap = (min(maxX, xLen) - max(minX, 0) )
    zOverlap = (min(maxZ, zLen) - max(minZ, 0) )
    if xOverlap < 0 or zOverlap < 0:
        return False

    areaTotal = (maxX - minX) * (maxZ - minZ )
    areaOverlap = xOverlap * zOverlap
    if areaOverlap / areaTotal > 0.7:
        return True
    else:
        return False


def findSupport(lverts, boxes, cats ):


    # Find support for every object
    boxList = []
    for n in range(0, len(boxes) ):
        bList = []
        top = boxes[n][0][:, 1].max()

        for m in range(0, len(boxes ) ):
            if m != n:
                bverts = boxes[m][0]
                minY, maxY = bverts[:, 1].min(), bverts[:, 1].max()

                bottom = minY
                if np.abs(top - bottom) < 0.75 * (maxY - minY ) and np.abs(top - bottom ) < 1:
                    isOverlap = checkOverlapApproximate(boxes[n][0], boxes[m][0] )
                    if isOverlap:
                        if m < n:
                            if not n in boxList[m]:
                                bList.append(m )
                        else:
                            bList.append(m )
        boxList.append(bList )


    # Find objects on floor
    floorList = []
    floorHeight = lverts[:, 1].min()
    for n in range(0, len(boxes ) ):
        isSupported = False
        for bList in boxList:
            if n in bList:
                isSupported = True
                break

        if not isSupported:
            if cats[n] == '03046257' or cats[n] == '03636649' or cats[n] == '02808440':
                bverts = boxes[n][0]
                minY, maxY = bverts[:, 1].min(), bverts[:, 1].max()
                if np.abs(minY - floorHeight ) < 1.5 * (maxY - minY) and np.abs(minY - floorHeight ) < 1 :
                    floorList.append(n )
            else:
                floorList.append(n )

    return floorList, boxList


def adjustHeightBoxes(boxId, boxes, cads, boxList ):
    top = boxes[boxId ][0][:, 1].max()
    for n in boxList[boxId ]:
        bverts = boxes[n][0]
        bottom = bverts[:, 1].min()
        delta = np.array([0, top-bottom, 0] ).reshape(1, 3)

        boxes[n][0] = boxes[n][0] + delta
        cads[n][0] = cads[n][0] + delta

        boxes[n].append( ('t', delta.squeeze() ) )
        cads[n].append( ('t', delta.squeeze() ) )
        if len(boxList[n]) != 0:
            adjustHeightBoxes(n, boxes, cads, boxList )
            adjustHeightBoxes(n, boxes, cads, boxList )
    return


def adjustHeight(lverts, boxes, cads, floorList, boxList ):
    # Adjust the height
    floorHeight = lverts[:, 1].min()
    for n in floorList:
        bverts = boxes[n][0]
        bottom = bverts[:, 1].min()
        delta = np.array([0, floorHeight-bottom, 0] ).reshape(1, 3)

        boxes[n][0] = boxes[n][0] + delta
        boxes[n].append( ('t', delta.squeeze() ) )
        cads[n][0] = cads[n][0] + delta
        cads[n].append( ('t', delta.squeeze() ) )

        if len(boxList[n] ) != 0:
            adjustHeightBoxes(n, boxes, cads, boxList )

    return


def checkPointInPolygon(wallVertices, v ):
    ###Given the wall vertices, determine if the pt is inside the polygon
    X = [pt[0] for pt in wallVertices ]
    Z = [pt[2] for pt in wallVertices ]
    j = len(wallVertices) - 1

    oddNodes = False
    x, z = v[0], v[2]
    for i in range(len(wallVertices ) ):
        if Z[i] < z and Z[j] >= z or Z[j] < z and Z[i] >= z:
            if (X[i] + ((z - Z[i]) / (Z[j] - Z[i]) * (X[j] - X[i]) ) ) < x:
                oddNodes = not oddNodes
        j=i
    return oddNodes


def calLineParam(pt1, pt2 ):
    ###Calculate line parameters
    x1, z1 = pt1
    x2, z2 = pt2

    a = z1 - z2
    b = x2 - x1
    c = z2 * x1 - x2 * z1
    return a, b, c


def findNearestPt(w1, w2, pts ):
    ###Find the nearest point on the line to a point
    a, b, c = calLineParam(w1, w2)
    x, z = pts
    a2b2 = a ** 2 + b ** 2
    new_x = (b * (b * x - a * z) - a * c) / a2b2
    new_z = (a * (-b * x + a * z) - b * c) / a2b2
    return np.array([new_x, new_z] )


def findNearestWall(pt, wallVertices ):
    ###Find nearest wall of a point
    minD, result = 100, None
    pt = np.array([pt[0], pt[2]], dtype=np.float32 )
    j = len(wallVertices) - 1
    for i in range(len(wallVertices ) ):
        w1 = np.array([wallVertices[i][0], wallVertices[i][2] ], dtype = np.float32 )
        w2 = np.array([wallVertices[j][0], wallVertices[j][2] ], dtype = np.float32 )
        if np.linalg.norm(w1 - pt ) < np.linalg.norm(w2 - pt):
            d = np.linalg.norm(np.cross(w2 - w1, w1 - pt) ) / np.linalg.norm(w2 - w1)
        else:
            d = np.linalg.norm(np.cross(w2 - w1, w2 - pt) ) / np.linalg.norm(w2 - w1)
        if d < minD:
            nearestPt = findNearestPt(w1, w2, pt)
            denom, nom  = w1 - w2, w1 - nearestPt
            if(np.sum(denom == 0)):
                denom[denom == 0] = denom[denom != 0]
            check = nom / denom
            if np.mean(check) < 1 and np.mean(check) > 0:
                minD = d
                result = nearestPt
        j = i

    for i in range(len(wallVertices ) ):
        w1 = np.array([wallVertices[i][0], wallVertices[i][2] ], dtype = np.float32 )
        d = np.linalg.norm(w1 - pt)
        if d < minD:
            minD = d
            result = w1
    return minD, result


def moveBox(record):
    pt, nearestPt = record
    vector = ((nearestPt[0] - pt[0]), (nearestPt[1] - pt[2] ) )
    return vector

def moveBoxInWall(cverts, bboxes, cads, threshold = 0.3):
    # find wall_vertices
    wallVertices = []
    floorHeight = cverts[:, 1].min()
    for n in range(0, cverts.shape[0] ):
        vert = cverts[n, :]
        if np.abs(vert[1] - floorHeight ) < 0.1:
            wallVertices.append(vert )

    isMove = False
    isBeyondRange = False
    for n in range(0, len(boxes ) ):
        box = boxes[n]
        maxD, record = 0, None
        bverts = box[0]
        for m in range(0, bverts.shape[0] ):
            v = bverts[m, :]
            if not checkPointInPolygon(wallVertices, v ):
                d, nearestPt = findNearestWall(v, wallVertices )
                if maxD < d:
                    record = (v, nearestPt )
                    maxD = d

        if record != None:
            t_x, t_z = moveBox(record )
            trans = np.array([t_x, 0, t_z], dtype=np.float32 )
            if np.linalg.norm(trans ) > threshold:
                isBeyondRange = True
            if np.linalg.norm(trans ) >= 1e-7:
                isMove = True
                direc = trans / np.linalg.norm(trans )
                trans = trans + direc * 0.04

                bboxes[n][0] = bboxes[n][0] + trans.reshape(1, 3)
                bboxes[n].append( ('t', trans.squeeze() ) )

                cads[n][0] = cads[n][0] + trans.reshape(1, 3)
                cads[n].append( ('t', trans.squeeze() ) )

    return isMove, isBeyondRange


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default="./xml/", help="outDir of xml file" )
    parser.add_argument('--threshold', type=float, default = 0.3, help = 'the threshold to decide low quality mesh.')
    parser.add_argument('--rs', type=int, default=0, help='the starting point')
    parser.add_argument('--re', type=int, default=1600, help='the end point')
    opt = parser.parse_args()

    params = JSONHelper.read("./Parameters.json" )

    filename_json = params["scan2cad"]
    shapeNetRoot = params["shapenetAbs"]
    layoutRoot = params["scannet_layoutAbs"]

    with open('lowQualLayout.txt', 'w') as fOut:
        fOut.write('#Low quality layouts\n')

    sceneCnt = 0
    for r in JSONHelper.read(filename_json ):
        if not (sceneCnt >= opt.rs and sceneCnt < opt.re):
            continue
        sceneCnt += 1

        id_scan = r["id_scan"]
        print('%d: %s' % (sceneCnt, id_scan ) )

        outDir = osp.abspath(opt.out + "/" + id_scan )
        os.system('mkdir -p %s' % outDir )

        layOutFile = osp.join(layoutRoot, id_scan, id_scan + '.obj')
        contourFile = osp.join(layoutRoot, id_scan, id_scan + '_contour.obj')
        t = r['trs']['translation']
        q = r['trs']['rotation']
        s = r['trs']['scale']

        lverts, lfaces = loadMesh(layOutFile )
        lverts[:, 0], lverts[:, 1] = lverts[:, 0], lverts[:, 1]
        lverts, trans, rot, scale = computeTransform(lverts, t, q, s )
        layout = [lverts, lfaces, ('s', scale), ('rot', rot), ('t', trans) ]

        cverts, cfaces = loadMesh(contourFile )
        cverts, trans, rot, scale = computeTransform(cverts, t, q, s )

        boxes = []
        cads = []
        cats = []
        # Load the shapes
        for model in r["aligned_models"]:
            t = model["trs"]["translation"]
            q = model["trs"]["rotation"]
            s = model["trs"]["scale"]

            id_cad = model["id_cad"]
            catid_cad = model["catid_cad"]

            cad_file = osp.join(shapeNetRoot, catid_cad, id_cad, 'alignedNew.obj' )
            if not osp.isfile(cad_file ):
                continue

            vertices, faces = loadMesh(cad_file )
            bverts, bfaces = computeBox(vertices )

            bverts, trans, rot, scale = computeTransform(bverts, t, q, s )
            vertices, _, _, _ = computeTransform(vertices, t, q, s )

            boxes.append([bverts, bfaces, ('s', scale), ('rot', rot), ('t', trans) ] )
            cads.append([vertices, faces, ('s', scale), ('rot', rot), ('t', trans) ] )

            cats.append(catid_cad )

        # Output origin bounding boxes
        sceneDir = osp.join(outDir, 'scenBoxes' )
        os.system('mkdir %s' % sceneDir )

        sceneOrigName = osp.join(sceneDir, 'boxesOrigin.obj' )
        sceneBoxes = boxes + [layout ]
        writeScene(sceneOrigName, sceneBoxes )

        sceneOrigName = osp.join(sceneDir, 'cadsOrigin.obj' )
        sceneCads = cads + [layout ]
        writeScene(sceneOrigName, sceneCads )

        # Build the relationship and adjust heights
        floorList, boxList = findSupport(lverts, boxes, cats )
        adjustHeight(lverts, boxes, cads, floorList, boxList )

        # Push the boxes to be inside the room
        isMove, isBeyondRange = moveBoxInWall(cverts, boxes, cads, opt.threshold )
        cnt = 0
        while isMove == True and isBeyondRange == False:
            isMove, isBeyondRange = moveBoxInWall(cverts, boxes, cads, opt.threshold )
            print('IterNum %d' % cnt )
            cnt += 1
            if cnt == 5 or isMove == False or isBeyondRange == True:
                break

        if isBeyondRange == True:
            with open('lowQualLayout.txt', 'a') as fOut:
                fOut.write('%s\n' % id_scan )
        else:
            # Write new boxes
            sceneNewName = osp.join(sceneDir, 'boxesNew.obj')
            sceneBoxes = boxes + [layout ]
            writeScene(sceneNewName, sceneBoxes )

            sceneNewName = osp.join(sceneDir, 'cadsNew.obj')
            sceneCads = cads + [layout ]
            writeScene(sceneNewName, sceneCads )

            # Write the transformation
            '''
            transformList = []
            transformList.append(layout[2:] )
            for box in boxes:
                transformList.append(box[2:] )
            transformName = osp.join(outDir, 'transform.dat')

            with open(transformName, 'wb') as fOut:
                pickle.dump(transformList, fOut )
            '''
