# Copyright (C) 2024  MPI IS, Marilyn Keller
import argparse
import os
import pickle
import numpy as np

from skel.alignment.aligner import SkelFitter, load_smpl_seq


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Align SKEL to a SMPL sequence')
    
    # parser.add_argument('smpl_seq_path', type=str, help='Path to the SMPL sequence')
    parser.add_argument('-o', '--out_dir', type=str, help='Output directory', default='output')
    parser.add_argument('-F', '--force-recompute', help='Force recomputation of the alignment', action='store_true')
    parser.add_argument('-D', '--debug', help='Only run the fit on the first minibach to test', action='store_true')
    parser.add_argument('-B', '--batch-size', type=int, help='Batch size', default=3000)
    parser.add_argument('-w', '--watch-frame', type=int, help='Frame of the batch to display', default=0)
    
    args = parser.parse_args()
    
    # smpl_seq = load_smpl_seq(args.smpl_seq_path)
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
        "gender": 'male'
    }

    # 将单帧数据扩展到多帧，假设扩展到2帧
    n_frames = 2
    pose_glb = np.tile(data['pose_glb'], (n_frames, 1))
    pose_pca = np.tile(data['pose_pca'], (n_frames, 1))
    trans_glb = np.tile(data['trans_glb'], (n_frames, 1))
    betas = np.tile(data['shape'], (n_frames, 1))
    gender = data['gender']

    # 补齐poses数据到72个元素
    pose_full = np.zeros((n_frames, 72))
    pose_full[:, :3] = pose_glb
    pose_full[:, 3:72] = pose_pca

    # 构建符合程序要求的字典
    smpl_seq = {
        'trans': trans_glb,
        'poses': pose_full,
        'betas': betas,
        'gender': gender
    }
    
    # outfile = os.path.basename(args.smpl_seq_path).split(".")[0] + '_skel.pkl'
    outfile = 'smpl_w_skel.pkl'
    out_path = os.path.join(args.out_dir, outfile)
    
    if os.path.exists(out_path):
        print('Previous aligned SKEL sequence found at {}. Will be used as initialization.'.format(out_path))
        skel_data_init = pickle.load(open(out_path, 'rb'))
    else:
        skel_data_init = None
    
    skel_fitter = SkelFitter(smpl_seq['gender'], device='cuda:0')
    skel_seq = skel_fitter.fit(smpl_seq['trans'], 
                               smpl_seq['betas'], 
                               smpl_seq['poses'], 
                               batch_size = args.batch_size,
                               skel_data_init=skel_data_init, 
                               force_recompute=args.force_recompute,
                               debug=args.debug,
                               watch_frame=args.watch_frame)
    
    pickle.dump(skel_seq, open(out_path, 'wb'))
    print('Saved aligned SKEL sequence to {}'.format(out_path))
