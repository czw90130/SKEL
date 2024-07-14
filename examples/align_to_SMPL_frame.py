# Copyright (C) 2024  MPI IS, Marilyn Keller
import argparse
import os
import pickle

import trimesh
from skel.alignment.aligner import SkelFitter
from skel.alignment.utils import load_smpl_seq


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Align SKEL to a SMPL frame')
    
    parser.add_argument('--smpl_mesh_path', type=str, help='Path to the SMPL mesh to align to', default=None)
    parser.add_argument('--smpl_data_path', type=str, help='Path to the SMPL dictionary to align to (.pkl or .npz)', default=None)
    parser.add_argument('-o', '--out_dir', type=str, help='Output directory', default='output')
    parser.add_argument('-F', '--force-recompute', help='Force recomputation of the alignment', action='store_true')
    parser.add_argument('--gender', type=str, help='Gender of the subject (only needed if not provided with smpl_data_path)', default='female')
    
    args = parser.parse_args()
    
    if args.smpl_data_path is None:
        import numpy as np
        # 假设你提供的数据（单帧）
        data = {
            "pose_glb": np.array([0.0830066, 0.00511488, 0.01239173]),
            "pose_pca": np.array([-0.10971282,  0.03151396, -0.02468906, -0.12673404, -0.04046477, -0.00999856,
                                0.10337554,  0.00701611, -0.01237919,  0.04228049,  0.40904755,  0.0103508,
                                0.06761938, -0.38678374, -0.02457267, -0.05254,     0.01622367,  0.00543508,
                                0.,          0.,          0.,          0.,          0.,          0.,
                                -0.22898743, -0.03359564, -0.01303963,  0.,          0.,          0.,
                                0.,          0.,          0.,          0.,          0.,          0.,
                                0.06987433,  0.25087804, -0.62984411,  0.03026429, -0.23722501,  0.61997123,
                                0.,          0.,          0.,         -0.1509841,  -0.20050173, -0.82868125,
                                -0.02511523,  0.18954968,  0.83344669, -0.91029704, -0.13552834, -0.06455877,
                                -1.01220866,  0.13567254,  0.08596888,  0.,          0.,          0.,
                                0.,          0.,          0.,          0.,          0.,          0.,
                                0.,          0.,          0.]),
            "shape": np.array([ 5.75246826,  0.09169463,  2.59739513, -5.06456705,  0.06703315,  0.18366071,
                            -0.92888224, -1.53696401,  0.72576201,  5.92534986]),
            "trans_glb": np.array([ 0.00243988,  1.19291523, -0.04295483]),
            "scale_glb": 1.2498714758854579,
            "gender": 'm'
        }

        # 将单帧数据扩展到多帧，假设扩展到10帧
        n_frames = 1
        pose_glb = np.tile(data['pose_glb'], (n_frames, 1))
        pose_pca = np.tile(data['pose_pca'], (n_frames, 1))
        trans_glb = np.tile(data['trans_glb'], (n_frames, 1))
        betas = np.tile(data['shape'], (n_frames, 1))

        # 补齐poses数据到72个元素
        pose_full = np.zeros((n_frames, 72))
        pose_full[:, :3] = pose_glb
        pose_full[:, 3:72] = pose_pca

        # 构建符合程序要求的字典
        smpl_data = {
            'trans': trans_glb,
            'poses': pose_full,
            'betas': betas,
            'gender': args.gender
        }

    else:
        smpl_data = load_smpl_seq(args.smpl_data_path, gender=args.gender, straighten_hands=False)
        pass
    
    if args.smpl_mesh_path is not None:
        subj_name = os.path.basename(args.smpl_mesh_path).split(".")[0]
    elif args.smpl_data_path is not None:
        subj_name = os.path.basename(args.smpl_data_path).split(".")[0]
    else:
        raise ValueError('Either smpl_mesh_path or smpl_data_path must be provided')
    
    # Create the output directory
    subj_dir = os.path.join(args.out_dir, subj_name)
    os.makedirs(subj_dir, exist_ok=True)
    pkl_path = os.path.join(subj_dir, subj_name+'_skel.pkl')  
    
    subj_dir = subj_dir
    
    if os.path.exists(pkl_path) and not args.force_recompute:
        print('Previous aligned SKEL sequence found at {}. Will be used as initialization.'.format(subj_dir))
        skel_data_init = pickle.load(open(pkl_path, 'rb'))
    else:
        skel_data_init = None
    
    skel_fitter = SkelFitter(smpl_data['gender'], device='cuda:0', export_meshes=True)
    skel_seq = skel_fitter.run_fit(smpl_data['trans'], 
                               smpl_data['betas'], 
                               smpl_data['poses'],
                               batch_size=1,
                               skel_data_init=skel_data_init, 
                               force_recompute=args.force_recompute)
    
    print('Saved aligned SKEL to {}'.format(subj_dir))
    
    SKEL_skin_mesh = trimesh.Trimesh(vertices=skel_seq['skin_v'][0], faces=skel_seq['skin_f'])
    SKEL_skel_mesh = trimesh.Trimesh(vertices=skel_seq['skel_v'][0], faces=skel_seq['skel_f'])
    SMPL_mesh = trimesh.Trimesh(vertices=skel_seq['smpl_v'][0], faces=skel_seq['smpl_f'])
    
    SKEL_skin_mesh.export(os.path.join(args.out_dir, subj_name + '_skin.obj'))
    SKEL_skel_mesh.export(os.path.join(args.out_dir, subj_name + '_skel.obj'))
    SMPL_mesh.export(os.path.join(args.out_dir, subj_name + '_smpl.obj'))
    
    pickle.dump(skel_seq, open(pkl_path, 'wb'))
    
    print('SKEL parameters saved to {}'.format(subj_dir))
    print('SKEL skin mesh saved to {}'.format(os.path.join(subj_dir, subj_name + '_skin.obj')))
    print('SKEL skel mesh saved to {}'.format(os.path.join(subj_dir, subj_name + '_skel.obj')))
    print('SMPL mesh saved to {}'.format(os.path.join(subj_dir, subj_name + '_smpl.obj')))
    