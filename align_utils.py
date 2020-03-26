import os
import copy
import random
import numpy as np
import open3d as o3d
from skimage import transform
from scipy.spatial.transform import Rotation as R

import pdb

def normalize(v):
    v_len = np.linalg.norm(v)
    ret = v
    if v_len != 0:
        ret /= v_len

    return ret

def get_light_corners(light_ctr, obj_corners):
    """Generate the four corners of the light given the corners of the .obj file
    Arguments:
        light_ctr: is equivalent to the entry 'light_ctr',
                        provided in scene????_??_corners.npy
        obj_corners: the annotated corners of individual objects, stored in bbox.txt
    Returns:
        mesh_corners: the four corners in 3D room, ordered as lower_left, lower_right, upper_right, upper_left
    """
    cx, cy, cz, cw = light_ctr[0, :]
    sides = np.ptp(obj_corners, axis=0)
    w, h = sides[sides > 0]
    scale = cw / w
    w *= scale
    h *= scale
    half_w, half_h = w/2, h/2
    lower_left =  np.array([cx-half_w, cy-half_h, cz])
    lower_right = np.array([cx+half_w, cy-half_h, cz])
    upper_right = np.array([cx+half_w, cy+half_h, cz])
    upper_left =  np.array([cx-half_w, cy+half_h, cz])
    mesh_corners = np.vstack([lower_left, lower_right, upper_right, upper_left])

    return mesh_corners

def get_transform(mesh_corners, obj_corners, obj_type, offset=0.1):
    """Estimates the transformation parameters, the ordered to be applied is
        scale -> rotate -> translate

    Arguments:
        mesh_corners:
            the four corners in 3D room ordered as lower_left, lower_right, upper_right, upper_left
            numpy.ndarray of shape [4, 3]
        obj_corners:
            the annotated corners of individual objects, ordered as lower_left, lower_right, upper_right, upper_left,
                stored in bbox.txt, numpy.ndarray of shape [4, 3]
        obj_type: 'w': window, 'd': door, 'c': curtain, 'li': light
        offset: offset in meters between curtain and wall
    """

    # make the object slightly smaller to ensure it covers the entire hole
    if obj_type == 'c':
        # calculate normal direction
        v = normalize(mesh_corners[1, :] - mesh_corners[0, :])
        up = np.array([0, 0, 1])
        N = np.cross(v, up)
        N = normalize(N)
        mesh_corners += offset * N

    # find scale factor
    # since the objects are axis-aligned, scaling factor can be found by calculating
    # the range of x, y, z dimensions
    obj_scales = np.ptp(obj_corners, axis=0)
    obj_scales[obj_scales == 0] = 1
    obj_scales = 1/obj_scales
    mesh_w = np.linalg.norm(mesh_corners[1, :] - mesh_corners[0, :])
    mesh_h = np.linalg.norm(mesh_corners[2, :] - mesh_corners[1, :])
    scales = (np.array([mesh_w, mesh_h, mesh_h]) + offset) * obj_scales
    scale_mat = np.diag(scales)

    scaled_corners = obj_corners @ scale_mat
    rigid_trans = transform.estimate_transform('euclidean', scaled_corners, mesh_corners)
    rigid_mat = rigid_trans.params
    t = rigid_mat[:3, 3]
    rot = rigid_mat[:3, :3]

    return [('s', scales), ('rot', rot), ('t', t)]

def retrieve_object(obj_folders, obj_type):
    objects = [f for f in os.listdir(obj_folders[obj_type]) if not '.' in f]
    obj = random.choice(objects)
    path = os.path.join(obj_folders[obj_type], obj, "{}")
    obj_mesh = o3d.io.read_triangle_mesh(path.format('merged.obj'))
    obj_corners = np.loadtxt(path.format('bbox.txt'))

    print('Type: {} - {}'.format(obj_type, obj))

    return obj_mesh, obj_corners

def generate_sample(mesh_root, obj_folders):
    scenes = [mdl.split('.')[0] for mdl in os.listdir(mesh_root) if mdl.endswith('.obj')]
    scn = random.choice(scenes)
    room_mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_root, scn+'.obj'))
    mesh_locs = np.load(os.path.join(mesh_root, scn+'_corners.npy'), allow_pickle=True).item()
    all_verts = np.asarray(room_mesh.vertices)
    all_tri = np.asarray(room_mesh.triangles)

    print('-'*30)
    print('Scene: {}'.format(scn))

    all_objects = [('li', mesh_locs['light_ctr'])] + list(zip(mesh_locs['type'], mesh_locs['coords']))
    for t, c in all_objects:
        lst = [t]
        if t == 'w': # if object is a window, also generate a curtain for it
            lst.append('c')

        for tt in lst:
            obj_mesh, obj_corners = retrieve_object(obj_folders, tt)
            obj_tri = np.asarray(obj_mesh.triangles)
            obj_verts = np.asarray(obj_mesh.vertices)
            obj_verts = np.vstack([obj_verts.T, np.ones([1, obj_verts.shape[0]])])

            if tt == 'li': # generate light corners on the fly
                c = get_light_corners(c, obj_corners)

            # estimate transform parameters
            T, T_params = get_transform(c, obj_corners, tt)
            new_verts = (T @ obj_verts).T[:, :3]
            new_tri = obj_tri + np.max(all_tri) + 1
            all_verts = np.vstack([all_verts, new_verts])
            all_tri = np.vstack([all_tri, new_tri])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(all_verts)
    mesh.triangles = o3d.utility.Vector3iVector(all_tri)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()

    print('-'*30)

    return mesh


if __name__ == "__main__":
    n_samples = 6
    obj_root = "../sample_data/meshes"
    mesh_root = "../Annotations/mesh_result"
    dump_root = "./samples"

    if not os.path.exists(dump_root):
        os.makedirs(dump_root)

    obj_folders = {
        'w': os.path.join(obj_root, 'window'),
        'd': os.path.join(obj_root, 'door'),
        'c': os.path.join(obj_root, 'curtain'),
        'li': os.path.join(obj_root, 'ceiling_lamp')
    }

    for i in range(n_samples):
        print('Mesh {}'.format(i+1))
        dpath = os.path.join(dump_root, 'tmp_{}.obj'.format(i+1))
        mesh = generate_sample(mesh_root, obj_folders)
        o3d.io.write_triangle_mesh(dpath, mesh)
