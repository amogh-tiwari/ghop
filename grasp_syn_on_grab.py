import trimesh
import numpy as np
import os
import os.path as osp
from preprocess.make_sdf_grid import get_sdf_grid
from tqdm import tqdm
import argparse

from utils.rot_utils import transform_meshes

import json
import numpy as np
from hydra import main
import hydra.utils as hydra_utils
import pickle
from glob import glob
import os.path as osp
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from jutils import (
    mesh_utils,
    image_utils,
    geom_utils,
    hand_utils,
    slurm_utils,
    web_utils,
)
from models.sd import SDLoss
from utils.obj2text import Obj2Text
from utils.io_util import load_sdf_grid
from utils.contact_util import compute_contact_loss

from grasp_syn import UniGuide

def get_base_meshes(meshes_dir, tgt_mesh_names=None):
    
    if tgt_mesh_names is None:
        tgt_mesh_names = [mesh_name.split("_")[0] for mesh_name in os.listdir(meshes_dir)]

    base_meshes = {}
    for mesh_name in tgt_mesh_names:
        mesh_fp = os.path.join(meshes_dir, mesh_name+".ply")
        mesh = trimesh.load(mesh_fp)
        base_meshes[mesh_name] = mesh
    
    return base_meshes

def validate_indices(start_idx, end_idx, target_indices, n_files):
    # Assert that both methods aren't used together
    using_range = start_idx is not None or end_idx is not None
    using_targets = target_indices is not None
    assert not (using_range and using_targets), "Cannot specify both (start_idx/end_idx) and target_indices together"
    
    if using_targets:
        # Convert to list if it's a string or other format
        if isinstance(target_indices, str):
            target_indices = [int(x.strip()) for x in target_indices.split(',')]
        
        # Validate indices
        for idx in target_indices:
            assert idx >= 0 and idx < n_files, f"target_indices contains invalid index {idx}. Must be in range [0, {n_files})"
        
        # Select only the target files
        sdf_list = [sdf_list[i] for i in target_indices]
        print(f"Selected {len(sdf_list)} files based on target_indices: {target_indices}")
    
    elif using_range:
        if end_idx is not None and end_idx > n_files:
            print(f"!!! WARNING !!! specified end_idx ({end_idx}) is greater than max number of files {n_files}. Setting end_idx to last idx ({n_files})")
            end_idx = n_files
        
        assert end_idx > start_idx, "end_idx must be greater than start_idx"
        assert end_idx >= 0 and start_idx >= 0, "both start_idx and end_idx must be non-negative"
        assert start_idx < n_files, "start_idx must be less than max number of elements"

def _main(args, custom_cfg):
    ##### ##### ##### Data and Pre-Loop Init ##### ##### #####
    dataset_dir = custom_cfg['dataset_dir']
    frames_fp = custom_cfg['frames_fp']
    params_fp = custom_cfg['params_fp']
    dst_dir_base = custom_cfg['dst_dir_base']
    all_frames_flag = custom_cfg['all_frames_flag']
    base_meshes_dir = custom_cfg['base_meshes_dir']
    # start_idx = custom_cfg['start_idx']
    # end_idx = custom_cfg['end_idx']
    device = "cuda"

    base_meshes = get_base_meshes(base_meshes_dir, tgt_mesh_names=['binoculars', 'camera', 'fryingpan', 'mug', 'toothpaste', 'wineglass'])
    frame_data = np.load(frames_fp)
    frame_names = frame_data['frame_names']
    n_frames = len(frame_names)

    # Add slicing based on start_idx and end_idx
    start_idx = args.get('start_idx', None)
    end_idx = args.get('end_idx', None)
    target_indices = args.get('target_indices', None)
    
    validate_indices(start_idx, end_idx, target_indices, n_frames)
    ### target_indices currently not supported. variable kept for legacy reasons.

    if all_frames_flag is False:
        frame_idxs = frame_data['selected_idxs'][start_idx:end_idx]
    else:
        frame_idxs = list(range(start_idx, end_idx))
    
    frame_params = np.load(params_fp)

    dst_dir = os.path.join(dst_dir_base, f"{start_idx:06d}_to_{end_idx:06d}/all")
    ##### ##### ##### Data and Pre-Loop Init ##### ##### #####


    ##### ##### ##### Parts of Batch Uniguide ##### ##### #####
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    uni_guide = UniGuide(args)
    base_dir = args.save_dir  #  "/home/yufeiy2/scratch/result/uni_guide/"

    sd = uni_guide.init_sds(args, device)

    ##### ##### ##### Parts of Batch Uniguide ##### ##### #####

    for f_name, f_idx in tqdm(zip(frame_names, frame_idxs), total=len(frame_names)):
        ##### ##### ##### Get Sample Info ##### ##### #####
        frame_info = np.load(os.path.join(dataset_dir, f_name))
        mesh_name = f_name.split("/")[-2].split("_")[0]
        base_mesh = base_meshes[mesh_name]
        rot_mats = frame_params['root_orient_obj_rotmat'][f_idx][0] # [0] to get (3,3) and not (1,3,3)
        trans_vecs = frame_params['trans_obj'][f_idx]
        transformed_meshes, transformed_verts, transformed_faces = transform_meshes([base_mesh], [rot_mats], [trans_vecs])
        name_base = "_".join(f_name.split("/")[2:5]).strip(".npz")
        seq_name = f"{f_idx:05d}_{name_base}"
        # dst_file = osp.join(dst_dir, f"{f_idx:05d}_{seq_name}", "oObj.obj")
        dst_file = osp.join(dst_dir, seq_name, "oObj.obj")
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)

        _ = transformed_meshes[0].export(dst_file) # [0] bcz it's a list of trimesh objects with only 1 element.
        verts = frame_info['verts_object']
        _ = trimesh.PointCloud(verts).export(dst_file.replace("oObj.obj", "oPcd.obj"))
        # os.system(f"cp {obj_file} {dst_file}")
        print(f"Saved stuff to {dst_dir}")
        ##### ##### ##### Get Sample Info ##### ##### #####

        ##### ##### ##### Get SDF Info ##### ##### #####
        # get sdf grid
        sdf_file = osp.join(dst_dir, seq_name, "uSdf.npz")
        sdf, transformation = get_sdf_grid(dst_file, 64, True)
        np.savez_compressed(sdf_file, sdf=sdf, transformation=transformation)

        # get text
        # breakpoint()
        text = seq_name.split("_")[2] # Get the object name --- The '[2]' is very seq / data specific.
        with open(osp.join(dst_dir, seq_name, "obj.txt"), "w") as f:
            f.write(text)
        # ---> At end of this block, you have sdf, transformation, text variables which shall be used later
        ##### ##### ##### Get SDF Info ##### ##### #####

        ##### ##### ##### Parts of Batch Uniguide ##### ##### #####
        if not args.get("sds_grasp", False): # if sds_grasp is False or not present, don't run below code.
            print("skipping sds_grasp")
            pass
        else:
            print("sds grasp")
            index = sdf_file.split("/")[-2]
            text, uSdf, nTu_fake, oTu, oObj_orig = uni_guide.read_one_grasp(sdf_file)
            cell_list = []

            web_file = osp.join(base_dir, f"{index}_grasp.html")
            for s in range(args.S):
                print(f"s={s}")
                bs = 1
                save_pref = osp.join(base_dir,  f"{index}_s{s:02d}")

                _, _, nTu_scale = geom_utils.homo_to_rt(nTu_fake)
                rot = geom_utils.random_rotations(bs, device=device)
                r = args.rand_tsl * 5  # in meter --> scale in normalized frame
                tsl = torch.rand([bs, 3], device=device) * r * 2 - r
                npTu = geom_utils.rt_to_homo(rot, tsl, s=nTu_scale)
                npSdf, _ = mesh_utils.transform_sdf_grid(uSdf, npTu, lim=1.5)

                nTnp, hA_pred, pred_loss = uni_guide.sds_grasp(
                    sd,
                    npSdf,
                    text,
                    cfg=args,
                    T=args.T,
                    save_pref=save_pref,
                    vis_every_n=args.vis_every_n
                )
                nTu = nTnp @ npTu

                uni_guide.save_grasp(nTu, hA_pred, oTu, oObj_orig, pred_loss, save_pref)
                line = [
                    pred_loss,
                    save_pref + f"_t{args.T:04d}_pred.gif",
                    save_pref + f"_t{0:04d}_pred.gif",
                ]
                cell_list.append(line)

            # sort row of cell_list by their 1st column
            metric_list = [e[0] for e in cell_list]
            # save metric list to json
            save_file = osp.join(osp.dirname(save_pref), f"{index}_metric.json")
            with open(save_file, "w") as f:
                json.dump(metric_list, f)
            idx = np.argsort(metric_list)
            sorted_cell_list = []
            for i in idx:
                sorted_cell_list.append(cell_list[i])

            sorted_cell_list.insert(
                0,
                [
                    "Final Loss",
                    "Final Grasp",
                    "Init Grasp",
                ],
            )
            web_utils.run(web_file, sorted_cell_list, width=256, inplace=True)

        if not args.get("refine_grasp", False): # if refine is False or not present, don't run below code.
            print("skipping refining")
            pass
        else:
            print("refining")
            index = sdf_file.split("/")[-2]
            text, uSdf, nTu_fake, oTu, oObj_orig = uni_guide.read_one_grasp(sdf_file)

            index = sdf_file.split("/")[-2]
            grasp_list = sorted(
                glob(osp.join(base_dir, f"{index}_*_para.pkl"))
            )
            for s, grasp_file in enumerate(grasp_list):
                basename = osp.basename(grasp_file).split("_para.pkl")[0]
                w_miss = args.loss.w_miss
                w_pen = args.loss.w_pen
                save_pref = osp.join(base_dir, "refine", basename)
                data = pickle.load(open(grasp_file, "rb"))
                nTu = torch.FloatTensor(data["nTu"]).to(device)[None]
                hA = torch.FloatTensor(data["hA"]).to(device)[None]

                nTu, hA = uni_guide.refine_grasp(
                    oObj_orig,
                    hA,
                    nTu,
                    oTu,
                    save_pref=save_pref,
                    w_pen=w_pen,
                    w_miss=w_miss,
                    vis_every_n=args.vis_every_n
                )

                uni_guide.save_grasp(nTu, hA, oTu, oObj_orig, data["loss"], save_pref)
        ##### ##### ##### Parts of Batch Uniguide ##### ##### #####


@main(config_path="configs", config_name="grasp_syn", version_base=None)
@slurm_utils.slurm_engine()
def main(hydra_cfg):
    ##### ----- ACCOUNT SPECIFIC ----- #####
    SCRATCH = "/scratch/clear/atiwari"
    # SCRATCH = "/lustre/fsn1/projects/rech/tuy/ulc52bd/"
    dst_dir_base = f"{SCRATCH}/datasets/grabnet_processing_all_sdfs/"
    ##### ----- ACCOUNT SPECIFIC ----- #####

    dataset_dir = f"{SCRATCH}/datasets/grabnet_extract/data"
    # frames_fp = "/scratch/clear/atiwari/datasets/grabnet_subset/data/test/frame_names_one_sample_per_sequence.npz"
    frames_fp = f"{SCRATCH}/datasets/grabnet_extract/data/test/frame_names.npz"
    params_fp = f"{SCRATCH}/datasets/grabnet_extract/data/test/grabnet_test.npz"
    all_frames_flag = True
    base_meshes_dir = f"{SCRATCH}/datasets/grabnet_extract/tools/object_meshes/contact_meshes/"

    if not all_frames_flag and "subset" in frames_fp:
        print("all_frames_flag is False but frames_fp has 'subset' in it's path. Exitting ...")
        exit()

    custom_cfg = {
        'dataset_dir': dataset_dir,
        'frames_fp': frames_fp,
        'params_fp': params_fp,
        'dst_dir_base': dst_dir_base,
        'all_frames_flag': all_frames_flag,
        'base_meshes_dir': base_meshes_dir,
    }

    _main(hydra_cfg, custom_cfg)


if __name__ == "__main__":
    main()
