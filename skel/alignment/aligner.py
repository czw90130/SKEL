
"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Soyong Shin, Marilyn Keller
See https://skel.is.tue.mpg.de/license.html for licensing and contact information.
"""

import argparse
import math
import os
import pickle
import torch
import numpy as np
from tqdm import trange
import smplx
import torch.nn.functional as F
from psbody.mesh import Mesh, MeshViewer
import skel.config as cg
from skel.skel_model import SKEL

def compute_scapula_loss(poses):
    """
    计算肩胛骨姿势的损失
    
    参数:
    poses: 姿势参数张量
    
    返回:
    scapula_loss: 肩胛骨姿势的L2范数损失
    """
    scapula_indices = [26, 27, 28, 36, 37, 38]
    
    scapula_poses = poses[:, scapula_indices]
    scapula_loss = torch.linalg.norm(scapula_poses, ord=2)
    return scapula_loss

def compute_spine_loss(poses):
    """
    计算脊柱姿势的损失
    
    参数:
    poses: 姿势参数张量
    
    返回:
    spine_loss: 脊柱姿势的L2范数损失
    """
    spine_indices = range(17, 25)  # 脊柱相关的关节索引
    
    spine_poses = poses[:, spine_indices]
    spine_loss = torch.linalg.norm(spine_poses, ord=2)
    return spine_loss

# 计算整体姿势损失
def compute_pose_loss(poses):
    """
    计算整体姿势的损失（不包括全局旋转）
    
    参数:
    poses: 姿势参数张量
    
    返回:
    pose_loss: 整体姿势的L2范数损失
    """
    pose_loss = torch.linalg.norm(poses[:, 3:], ord=2) # The global rotation should not be constrained 不约束全局旋转
    return pose_loss

# 计算时间连续性损失
def compute_time_loss(poses):
    """
    计算相邻帧之间姿势变化的损失
    
    参数:
    poses: 姿势参数张量
    
    返回:
    time_loss: 相邻帧姿势差异的L2范数损失
    """
    pose_delta = poses[1:] - poses[:-1]
    time_loss = torch.linalg.norm(pose_delta, ord=2)
    return time_loss

# 优化函数
def optim(params, 
          poses,
          betas,
          trans,
          verts,
          skel_model,
          device,
          lr=1e0,
          max_iter=5,
          num_steps=5,
          line_search_fn='strong_wolfe',
          rot_only=False,
          watch_frame=0,
          pose_reg_factor = 1e1,
          ):
        """
        优化SKEL模型参数以匹配目标顶点
        
        参数:
        params: 需要优化的参数列表
        poses: 姿势参数
        betas: 体型参数
        trans: 平移参数
        verts: 目标顶点
        skel_model: SKEL模型
        device: 计算设备
        lr: 学习率
        max_iter: 每步最大迭代次数
        num_steps: 优化步数
        line_search_fn: 线搜索函数
        rot_only: 是否只优化旋转
        watch_frame: 可视化的帧索引
        pose_reg_factor: 姿势正则化因子
        
        返回:
        无返回值，直接修改输入参数
        """
    
        # poseLoss = PoseLimitLoss().to(device)
        
        # regress joints 
        anat_joints = torch.einsum('bik,ji->bjk', [verts, skel_model.J_regressor_osim]) 
        fitting_mask_file = 'skel/alignment/riggid_parts_mask.pkl'
        fitting_indices = pickle.load(open(fitting_mask_file, 'rb'))
        fitting_mask = torch.zeros(6890, dtype=torch.float)
        fitting_mask[fitting_indices] = 1
        fitting_mask = fitting_mask.unsqueeze(0).to(device)
        dJ=torch.zeros((poses.shape[0], 24, 3), device=betas.device)
        
        optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=max_iter, line_search_fn=line_search_fn)
        pbar = trange(num_steps, leave=False)
        if('DISABLE_VIEWER' in os.environ):
            mv = None
            print("\n DISABLE_VIEWER flag is set, running in headless mode")
        else:
            mv = MeshViewer(keepalive=False)

        def closure():
            optimizer.zero_grad()
            fi = watch_frame #frame of the batch to display
            output = skel_model.forward(poses=poses[fi:fi+1], 
                                        betas=betas[fi:fi+1], 
                                        trans=trans[fi:fi+1], 
                                        poses_type='skel', 
                                        dJ=dJ[fi:fi+1],
                                        skelmesh=True)
            meshes_to_display = [Mesh(v=output.skin_verts[0].detach().cpu().numpy(), f=[], vc='white')] \
                    + [Mesh(v=verts[fi].detach().cpu().numpy(), f=[], vc='green')] \
                    + [Mesh(v=output.skel_verts[0].detach().cpu().numpy(), f=skel_model.skel_f.cpu().numpy(), vc='white')] \
                    # + location_to_spheres(output.joints.detach().cpu().numpy()[frame_to_watch], color=(1,0,0), radius=0.02)
                    # + [Mesh(v=verts[frame_to_watch].detach().cpu().numpy(), f=smpl_data.faces, vc='green')] \
                    # + location_to_spheres(joints_reg[frame_to_watch].detach().cpu().numpy(), color=(0,1,0), radius=0.02) \
            if('DISABLE_VIEWER' not in os.environ):
                mv.set_dynamic_meshes(meshes_to_display)
            
            # print(poses[frame_to_watch, :3])
            # print(trans[frame_to_watch])
            # print(betas[frame_to_watch, :3])
            # mv.get_keypress()
            
            if rot_only:
                mask = torch.zeros_like(poses).to(device)
                mask[:,:3] = 1
                poses_in = poses * mask
            else:
                poses_in = poses
            
            output = skel_model.forward(poses=poses_in, betas=betas, trans=trans, poses_type='skel', dJ=dJ, skelmesh=False)
            
            # Fit the SMPL vertices
            # We know the skinning of the forearm and the neck are not perfect,
            # so we create a mask of the SMPL vertices that are important to fit, like the hands and the head
            verts_loss_close = 1e3 * (fitting_mask.reshape(1, -1, 1) * (output.skin_verts - verts)**2).mean()
            verts_loss_loose = 5e2 * ((output.skin_verts - verts)**2).mean()
            verts_loss = verts_loss_close + verts_loss_loose
            
            # Fit the regressed joints
            joint_loss = 1e2 * F.mse_loss(output.joints, anat_joints)
            
            # Regularize the pose
            scapula_loss = 1e-2 * compute_scapula_loss(poses_in)
            spine_loss = 1e-3 * compute_spine_loss(poses_in)
            pose_loss = 1e-4 * compute_pose_loss(poses)
            
            # Time consistancy
            time_loss = 1e-2 * compute_time_loss(poses)
            
            for pl in [scapula_loss, spine_loss, pose_loss]:
                pl = pose_reg_factor * pl
            
            loss = verts_loss + joint_loss + scapula_loss + spine_loss + pose_loss + time_loss
            # make a pretty print of the losses
            print(f"Verts loss: {verts_loss.item():.4f},  \
                  Joint loss: {joint_loss.item():.4f}, \
                  Scapula loss: {scapula_loss.item():.4f},\
                  Spine loss: {spine_loss.item():.4f}, \
                  Pose loss: {pose_loss.item():.4f}, \
                  Time loss: {time_loss.item():.4f}")
      
            loss.backward()
        
            return loss

        for _ in pbar:
            loss = optimizer.step(closure).item()
            with torch.no_grad():
                poses[:] = torch.atan2(poses.sin(), poses.cos())
            pbar.set_postfix_str(f"Loss {loss:.4f}")
            
# SKEL拟合器类
class SkelFitter(object):
    """
    用于将SKEL模型拟合到SMPL序列的类
    """
    def __init__(self, gender, device, num_betas=10) -> None:
        """
        初始化SkelFitter
        
        参数:
        gender: 性别 ('male' 或 'female')
        device: 计算设备
        num_betas: beta参数的数量
        """
        self.smpl = smplx.create(cg.smpl_folder, model_type='smpl', gender=gender, num_betas=num_betas, batch_size=1).to(device)
        self.skel = SKEL(gender).to(device)
        self.gender = gender
        self.device = device
        self.num_betas = num_betas
        
    def fit(self, trans_in, betas_in, poses_in, batch_size=20, skel_data_init=None, 
            force_recompute=False, 
            debug=False,
            watch_frame=0,
            freevert_mesh=None):
        """Align SKEL to a SMPL sequence.

        将SKEL模型拟合到SMPL序列
        
        参数:
        trans_in: 输入的平移参数
        betas_in: 输入的体型参数
        poses_in: 输入的姿势参数
        batch_size: 批处理大小
        skel_data_init: 初始SKEL数据（如果有）
        force_recompute: 是否强制重新计算
        debug: 是否开启调试模式
        watch_frame: 可视化的帧索引
        freevert_mesh: 自由顶点网格（如果有）
        
        返回:
        res_dict: 包含拟合结果的字典
        """

        # Optimization params
        init_optim_params = {'lr': 1e0, 'max_iter': 25, 'num_steps': 10}
        seq_optim_params = {'lr': 1e-1, 'max_iter': 10, 'num_steps': 10}
        
        nb_frames = poses_in.shape[0]
        print('Fitting {} frames'.format(nb_frames))
        
        if skel_data_init is None or force_recompute:
        
            poses_skel = np.zeros((nb_frames, self.skel.num_q_params))
            poses_skel[:, :3] = poses_in[:, :3].copy() # Global orient are similar between SMPL and SKEL, so init with SMPL angles
            
            betas_skel = np.zeros((nb_frames, 10)); 
            betas_skel[:] = betas_in[..., :10].copy()
            # betas_out = smpl_data.betas[..., :10].reshape(-1, 10).expand(nb_frames, -1).detach().cpu().numpy()
            
            trans_skel = trans_in.copy() # Translation is similar between SMPL and SKEL, so init with SMPL translation
            
        else:
            # Load from previous alignment
            poses_skel = skel_data_init['poses']
            betas_skel = skel_data_init['betas']
            trans_skel = skel_data_init['trans']
        
        # We cut the whole sequence in batches for parallel optimization
        
        if batch_size > nb_frames:
            batch_size = nb_frames
            print('Batch size is larger than the number of frames. Setting batch size to {}'.format(batch_size))
            
        n_batch = math.ceil(nb_frames/batch_size)
        pbar = trange(n_batch, desc='Running batch optimization')
        
        to_params = lambda x: torch.from_numpy(x).float().to(self.device).requires_grad_(True)
        to_torch = lambda x: torch.from_numpy(x).float().to(self.device)
        
        # initialize the res dict with generic empty lists 
        res_dict = {
            'poses':  [],
            'betas':  [],
            'trans':  [],
            'gender':  [],
        }

            
        
        for i in pbar:
            
            if debug:
                # Only run the first batch to test, ignore the rest
                if i > 1:
                    continue
            
            # Get mini batch
            i_start, i_end = i * batch_size, min((i+1) * batch_size, nb_frames)
            
            # SMPL params
            poses_smpl = to_torch(poses_in[i_start:i_end].copy())
            betas_smpl = to_torch(betas_in[:self.num_betas].copy()).expand(i_end-i_start, -1)
            trans_smpl = to_torch(trans_in[i_start:i_end].copy())
            
            # Run a SMPL forward pass to get the SMPL body vertices
            smpl_output = self.smpl(betas=betas_smpl, body_pose=poses_smpl[:,3:], transl=trans_smpl, global_orient=poses_smpl[:,:3])
            verts = smpl_output.vertices
            if(freevert_mesh is not None):
                verts = to_torch(freevert_mesh).unsqueeze(0).repeat_interleave(batch_size, 0)
            
            # SKEL params        
            poses = to_params(poses_skel[i_start:i_end].copy())
            betas = to_params(betas_skel[i_start:i_end].copy())
            trans = to_params(trans_skel[i_start:i_end].copy())
            
            if i == 0 and not skel_data_init:
                # Optimize the global rotation and translation for the initial fitting
                optim([trans,poses], poses, betas, trans, verts, self.skel, self.device, rot_only=True, watch_frame=watch_frame)
                optim([trans,poses], poses, betas, trans, verts, self.skel, self.device, watch_frame=watch_frame, **init_optim_params)
            else:
                optim([trans,poses], poses, betas, trans, verts, self.skel, self.device, watch_frame=watch_frame, pose_reg_factor=1, **seq_optim_params)
            
            # Save the result
            poses_skel[i_start:i_end] = poses[:].detach().cpu().numpy().copy()
            trans_skel[i_start:i_end] = trans[:].detach().cpu().numpy().copy()
            
            # Initialize the next frames with current frame
            poses_skel[i_end:] = poses[-1:].detach().cpu().numpy().copy()
            trans_skel[i_end:] = trans[-1].detach().cpu().numpy().copy()
            betas_skel[i_end:] = betas[-1:].detach().cpu().numpy().copy()

            res_dict['poses'].append(poses.detach().cpu().numpy().copy())
            res_dict['betas'].append(betas.detach().cpu().numpy().copy())
            res_dict['trans'].append(trans.detach().cpu().numpy().copy())
            res_dict['gender'] = self.gender
            res_dict['skel_f'] = self.skel.skel_f.cpu().numpy().copy()
            res_dict['skin_f'] = self.skel.skin_f.cpu().numpy().copy()
            
        for key, val in res_dict.items():
            if isinstance(val, list):
                res_dict[key] = np.concatenate(val)
                
        return res_dict
            
# 加载SMPL序列数据
def load_smpl_seq(smpl_seq_path):
    """
    从文件加载SMPL序列数据
    
    参数:
    smpl_seq_path: SMPL序列文件路径（.pkl 或 .npz）
    
    返回:
    out_dict: 包含SMPL数据的字典
    """
    if not os.path.exists(smpl_seq_path):
        raise Exception('Path does not exist: {}'.format(smpl_seq_path))
    
    if smpl_seq_path.endswith('.pkl'):
        data_dict = pickle.load(open(smpl_seq_path, 'rb'))
    
    elif smpl_seq_path.endswith('.npz'):
        data_dict = np.load(smpl_seq_path)
        data_dict = {key: data_dict[key] for key in data_dict.keys()} # convert to python dict
        
        # In some npz, the gender type happens to be: array('male', dtype='<U4'). So we convert it to string
        if not isinstance(data_dict['gender'], str):
            data_dict['gender'] = str(data_dict['gender'])
            
        if data_dict['poses'].shape[1] == 156:
            # Those are SMPL+H poses, we remove the hand poses to keep only the body poses
            poses = np.zeros((data_dict['poses'].shape[0], 72))
            poses[:, :72-2*3] = data_dict['poses'][:, :72-2*3] # We leave params for SMPL joints 22 and 23 to zero 
            data_dict['poses'] = poses
        
    for key in ['trans', 'poses', 'betas', 'gender']:
        assert key in data_dict.keys(), f'Could not find {key} in {smpl_seq_path}. Available keys: {data_dict.keys()})'
        
    out_dict = {}
    out_dict['trans'] = data_dict['trans']
    out_dict['poses'] = data_dict['poses']
    out_dict['betas'] = data_dict['betas']
    out_dict['gender'] = data_dict['gender']
    
    return out_dict