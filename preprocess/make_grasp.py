import numpy as np
import os
import os.path as osp

import argparse
from glob import glob

from preprocess.make_sdf_grid import get_sdf_grid

# folder/
#    object_0/
#        oObj.obj  // object mesh
#        uSdf.npz  // sdf grid
#        obj.txt   // class name

def _gather_benchmark_files(src_dir, name_template):
    object_names = os.listdir(src_dir)
    index_list = []
    object_files = []
    for object_name in object_names:
        object_dir = os.path.join(src_dir, object_name)
        if not os.path.isdir(object_dir):
            continue
        curr_object_files = glob(os.path.join(object_dir, f"*{name_template}*"))
        object_files.extend(curr_object_files)
        curr_index_names = [curr_object_file.replace(src_dir, "").replace(name_template, "").strip("/").replace("/", "_") for curr_object_file in curr_object_files]
        index_list.extend(curr_index_names)
    return index_list, object_files

def get_ho3d():
    # use dexYCB's npz
    src_dir = "/private/home/yufeiy2/scratch/data/DexYCB/"
    dst_dir = "/private/home/yufeiy2/scratch/result/GenGrasps/HO3D/"
    index_list = "003_cracker_box,006_mustard_bottle,011_banana,021_bleach_cleanser,035_power_drill,004_sugar_box,010_potted_meat_can,019_pitcher_base,025_mug,037_scissors".split(
        ","
    )
    obj_files = [osp.join(src_dir, "models", index, "textured_simple.obj") for index in index_list]
    return dst_dir, index_list, obj_files

def get_grab():
    src_dir = "/scratch/clear/atiwari/datasets/grabnet_extract/tools/object_meshes/contact_meshes"
    dst_dir = "./data/temp_grab_sdfs/meshes/all/"
    index_list = "binoculars,camera,fryingpan,mug,toothpaste,wineglass".split(",")
    obj_files = [osp.join(src_dir, index + ".ply") for index in index_list]
    return dst_dir, index_list, obj_files

def get_ho3d_benchmark():
    data_name = "ho3d_3000"
    name_template = "_mesh_transform.ply"
    src_dir = f"../object_manipulation/evaluation/benchmark/benchmarking/data/processed/transformed_assets/{data_name}"
    dst_dir = f"../object_manipulation/evaluation/benchmark/benchmarking/data/processed/ghop_prep/{data_name}"
    index_list, object_files = _gather_benchmark_files(src_dir, name_template)
    return dst_dir, index_list, object_files

def get_grab_benchmark():
    custom_data_name = "grab_3000"
    name_template = "_mesh_transform.ply"
    src_dir = f"../object_manipulation/evaluation/benchmark/benchmarking/data/processed/transformed_assets/{custom_data_name}"
    dst_dir = f"../object_manipulation/evaluation/benchmark/benchmarking/data/processed/ghop_prep/{custom_data_name}"
    index_list, object_files = _gather_benchmark_files(src_dir, name_template)
    return dst_dir, index_list, object_files

def get_custom(): # function might change as per requirements.
    raise NotImplementedError(f"Not yet implemented")


def get_data(data_name):
    if data_name == "ho3d":
        return get_ho3d()
    elif data_name == "grab":
        return get_grab()
    elif data_name == "ho3d_benchmark":
        return get_ho3d_benchmark()
    elif data_name == "grab_benchmark":
        return get_grab_benchmark()
    else:
        raise NotImplementedError(f"Don't know how to handle {data_name}")

def _set_range(start_idx, end_idx, n_samples):
    start_idx = max(0, start_idx)
    end_idx = n_samples if end_idx > n_samples else end_idx
    end_idx = start_idx+1 if end_idx == -1 else end_idx

    assert start_idx < end_idx, f"Invalid range: {start_idx} to {end_idx}"
    print(f"Processing {start_idx} to {end_idx} / {n_samples}")
    return start_idx, end_idx

def make_grasp(data_name, start_idx, end_idx):
    dst_dir, index_list, obj_files = get_data(data_name)
    start_idx, end_idx = _set_range(start_idx, end_idx, len(index_list))

    for idx in range(start_idx, end_idx):
        index = index_list[idx]
        obj_file = obj_files[idx]
        # # obj_file = osp.join(src_dir, "models", index, "textured_simple.obj")
        # # dst_file = osp.join(dst_dir, index, "oObj.obj")
        # obj_file = osp.join(src_dir, index + ".ply")
        ext = osp.splitext(obj_file)[1].lstrip(".")
        dst_file = osp.join(dst_dir, index, f"oObj.{ext}")

        os.makedirs(osp.dirname(dst_file), exist_ok=True)
        os.system(f"cp {obj_file} {dst_file}")
        print(f"copy {obj_file} to {dst_file}")

        # get sdf grid
        sdf_file = osp.join(dst_dir, index, "uSdf.npz")
        sdf, transformation = get_sdf_grid(dst_file, 64, True)
        np.savez_compressed(sdf_file, sdf=sdf, transformation=transformation)

        # get text
        text = index
        with open(osp.join(dst_dir, index, "obj.txt"), "w") as f:
            f.write(text)
    return


if __name__ == "__main__":
    # make_ho3d()
    # dataset_name = 'custom'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', required=True)
    parser.add_argument('--start_idx', default=-1, type=int, help="start index (0-based)")
    parser.add_argument('--end_idx', default=-1, type=int, help="end index (non inclusive)")
    args = parser.parse_args()
    
    make_grasp(args.data_name, args.start_idx, args.end_idx)
