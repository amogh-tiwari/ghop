import trimesh
import numpy as np
import os
import os.path as osp
from preprocess.make_sdf_grid import get_sdf_grid
from tqdm import tqdm
import argparse

import sys
sys.path.append("../")
from object_manipulation.utils.rot_utils import transform_meshes

# folder/
#    object_0/
#        oObj.obj  // object mesh
#        uSdf.npz  // sdf grid
#        obj.txt   // class name

def grab(base_meshes, dataset_dir, frames_fp, params_fp, dst_dir, all_frames_flag, start_idx, end_idx):
    frame_data = np.load(frames_fp)
    frame_names = frame_data['frame_names']
    if all_frames_flag is False:
        frame_idxs = frame_data['selected_idxs'][start_idx:end_idx]
    else:
        frame_idxs = list(range(start_idx, end_idx))
    
    frame_params = np.load(params_fp)

    for f_name, f_idx in tqdm(zip(frame_names, frame_idxs), total=len(frame_names)):
        frame_info = np.load(os.path.join(dataset_dir, f_name))
        mesh_name = f_name.split("/")[-2].split("_")[0]
        base_mesh = base_meshes[mesh_name]
        rot_mats = frame_params['root_orient_obj_rotmat'][f_idx][0] # [0] to get (3,3) and not (1,3,3)
        trans_vecs = frame_params['trans_obj'][f_idx]
        transformed_meshes, transformed_verts, transformed_faces = transform_meshes([base_mesh], [rot_mats], [trans_vecs])
        seq_name = "_".join(f_name.split("/")[2:4])
        dst_file = osp.join(dst_dir, seq_name, "oObj.obj")
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)

        _ = transformed_meshes[0].export(dst_file) # 0 bcz it's a list of trimesh objects with only 1 element.
        verts = frame_info['verts_object']
        _ = trimesh.PointCloud(verts).export(dst_file.replace("oObj.obj", "oPcd.obj"))
        # os.system(f"cp {obj_file} {dst_file}")
        print(f"Saved some stuff to {dst_file}")

        # get sdf grid
        sdf_file = osp.join(dst_dir, seq_name, "uSdf.npz")
        sdf, transformation = get_sdf_grid(dst_file, 64, True)
        np.savez_compressed(sdf_file, sdf=sdf, transformation=transformation)

        # get text
        text = seq_name.split("_")[1] # Get the object name
        with open(osp.join(dst_dir, seq_name, "obj.txt"), "w") as f:
            f.write(text)
    return

def get_base_meshes(meshes_dir, tgt_mesh_names=None):
    
    if tgt_mesh_names is None:
        tgt_mesh_names = [mesh_name.split("_")[0] for mesh_name in os.listdir(meshes_dir)]

    base_meshes = {}
    for mesh_name in tgt_mesh_names:
        mesh_fp = os.path.join(meshes_dir, mesh_name+".ply")
        mesh = trimesh.load(mesh_fp)
        base_meshes[mesh_name] = mesh
    
    return base_meshes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-idx', type=int, required=True)
    parser.add_argument('--end-idx', type=int, required=True)
    args = parser.parse_args()

    assert args.end_idx > args.start_idx, "end idx must be higher than start idx"
    assert args.start_idx >= 0 and args.end_idx >= 0, "both idxs must be non-negative"

    dataset_dir = "/scratch/clear/atiwari/datasets/grabnet_extract/data"
    # frames_fp = "/scratch/clear/atiwari/datasets/grabnet_subset/data/test/frame_names_one_sample_per_sequence.npz"
    frames_fp = "/scratch/clear/atiwari/datasets/grabnet_extract/data/test/frame_names.npz"
    params_fp = "/scratch/clear/atiwari/datasets/grabnet_extract/data/test/grabnet_test.npz"
    dst_dir = f"/scratch/clear/atiwari/datasets/grabnet_processing/sdfs_{args.start_idx:06d}_to_{args.end_idx:06d}/all"
    base_meshes_dir = "/scratch/clear/atiwari/datasets/grabnet_extract/tools/object_meshes/contact_meshes/"
    base_meshes = get_base_meshes(base_meshes_dir, tgt_mesh_names=['binoculars', 'camera', 'fryingpan', 'mug', 'toothpaste', 'wineglass'])

    grab(base_meshes, dataset_dir, frames_fp, params_fp, dst_dir, True, args.start_idx, args.end_idx) # The True is for all_frames_flag
