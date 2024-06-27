"""
Copyright©2024 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See https://skel.is.tue.mpg.de/license.html for licensing and contact information.
"""

import os
import torch.nn as nn
import torch
import numpy as np
import pickle as pkl
from typing import NewType, Optional

from skel.joints_def import curve_torch_3d, left_scapula, right_scapula
from skel.osim_rot import ConstantCurvatureJoint, CustomJoint, EllipsoidJoint, PinJoint, WalkerKnee
from skel.utils import build_homog_matrix, rotation_matrix_from_vectors, with_zeros, matmul_chain
from dataclasses import dataclass, fields

from skel.kin_skel import scaling_keypoints, pose_param_names, smpl_joint_corresp
import skel.config as cg

Tensor = NewType('Tensor', torch.Tensor)

@dataclass
class ModelOutput:
    """
    模型输出的数据类
    包含了模型输出的各种可选张量
    """
    vertices: Optional[Tensor] = None
    joints: Optional[Tensor] = None
    full_pose: Optional[Tensor] = None
    global_orient: Optional[Tensor] = None
    transl: Optional[Tensor] = None
    v_shaped: Optional[Tensor] = None

    def __getitem__(self, key):
        """
        获取指定键的值
        """
        return getattr(self, key)

    def get(self, key, default=None):
        """
        获取指定键的值，如果不存在则返回默认值
        """
        return getattr(self, key, default)

    def __iter__(self):
        """
        返回键的迭代器
        """
        return self.keys()

    def keys(self):
        """
        返回所有键的迭代器
        """
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        """
        返回所有值的迭代器
        """
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        """
        返回所有键值对的迭代器
        """
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)

@dataclass
class SKELOutput(ModelOutput):
    """
    SKEL模型的输出数据类，继承自ModelOutput
    包含了SKEL模型特有的输出字段
    
    1. body_pose:
   - 数据结构：Tensor
   - 维度：(B, 43)，其中B是批次大小
   - 用途：表示身体各个关节的局部旋转，不包括全局旋转
   - 关联：这是poses的一个子集，不包含全局旋转

    2. poses:
    - 数据结构：Tensor
    - 维度：(B, 46)，其中B是批次大小
    - 用途：表示完整的姿势参数，包括全局旋转和所有关节的局部旋转
    - 关联：包含body_pose，加上3个全局旋转参数

    3. joints:
    - 数据结构：Tensor
    - 维度：(B, Nj, 3)，其中B是批次大小，Nj是关节数量
    - 用途：表示变形后的关节在3D空间中的位置
    - 关联：由poses和betas共同决定，反映了姿势和形状变化后的关节位置

    4. joints_ori:
    - 数据结构：Tensor
    - 维度：(B, Nj, 3, 3)，其中B是批次大小，Nj是关节数量
    - 用途：表示每个关节的方向，通常是3x3旋转矩阵
    - 关联：由poses决定，反映了每个关节的旋转状态

    5. pose_offsets:
    - 数据结构：Tensor
    - 维度：(B, Ns, 3)，其中B是批次大小，Ns是顶点数量
    - 用途：表示由姿势变化引起的顶点偏移量
    - 关联：由poses计算得出，用于调整模型形状以适应不同的姿势

    这些参数之间的关系：

    - poses包含了完整的姿势信息，其中一部分（除去全局旋转）就是body_pose。
    - poses用于计算joints和joints_ori，决定了骨骼的位置和方向。
    - poses还用于计算pose_offsets，这些偏移量会应用到模型的顶点上，以实现更真实的姿势变形。
    - joints反映了最终的关节位置，综合了poses和betas的影响。
    - joints_ori反映了关节的方向，主要由poses决定。
    """
    betas: Optional[Tensor] = None # 形状参数,控制身体形状变化
    body_pose: Optional[Tensor] = None # 身体姿势参数,不包括全局旋转
    skin_verts: Optional[Tensor] = None  # 变形后的皮肤顶点位置
    skel_verts: Optional[Tensor] = None # 变形后的骨骼顶点位置
    joints: Optional[Tensor] = None # 变形后的关节位置
    joints_ori: Optional[Tensor] = None # 关节方向,通常表示为3x3旋转矩阵
    poses: Optional[Tensor] = None # 完整的姿势参数,包括全局旋转
    trans : Optional[Tensor] = None # 全局平移参数
    pose_offsets : Optional[Tensor] = None # 由姿势引起的顶点偏移量
    
    
class SKEL(nn.Module):
    """
    SKEL模型类
    实现了基于骨骼的人体模型
    """
    num_betas = 10
    
    def __init__(self, gender, model_path=None, **kwargs):
        """
        初始化SKEL模型
        
        参数：
        gender: 性别，'male'或'female'
        model_path: 模型文件路径，如果为None则使用默认路径
        """
        super(SKEL, self).__init__()

        if gender not in ['male', 'female']:
            raise RuntimeError(f'Invalid Gender, got {gender}')

        self.gender = gender
        
        if model_path is None:
            # skel_file = f"/Users/mkeller2/Data/skel_models_v1.0/skel_{gender}.pkl"
            skel_file = os.path.join(cg.skel_folder, f"skel_{gender}.pkl")
        else:
            skel_file = os.path.join(model_path, f"skel_{gender}.pkl")
        assert os.path.exists(skel_file), f"Skel model file {skel_file} does not exist"
        
        skel_data = pkl.load(open(skel_file, 'rb'))

        # 初始化模型参数
        self.num_betas = 10
        self.num_q_params = 46
        self.bone_names = skel_data['bone_names'] 
        self.num_joints = skel_data['J_regressor_osim'].shape[0]
        self.num_joints_smpl = skel_data['J_regressor'].shape[0]
        
        self.bone_axis = skel_data['bone_axis'] 
        self.joints_name = skel_data['joints_name']
        self.pose_params_name = skel_data['pose_params_name']
        
        # register the template meshes
        # 注册模板网格
        self.register_buffer('skin_template_v', torch.FloatTensor(skel_data['skin_template_v']))
        self.register_buffer('skin_f', torch.LongTensor(skel_data['skin_template_f']))
        
        self.register_buffer('skel_template_v', torch.FloatTensor(skel_data['skel_template_v']))
        self.register_buffer('skel_f', torch.LongTensor(skel_data['skel_template_f']))
        
        # Shape corrective blend shapes
        # 形状校正混合形状
        self.register_buffer('shapedirs', torch.FloatTensor(np.array(skel_data['shapedirs'][:,:,:self.num_betas])))
        self.register_buffer('posedirs', torch.FloatTensor(np.array(skel_data['posedirs'])))
        
        # Model sparse joints regressor, regresses joints location from a mesh
        # 模型稀疏关节回归器，从网格回归关节位置
        self.register_buffer('J_regressor', torch.FloatTensor(skel_data['J_regressor']))
        
        # Regress the anatomical joint location with a regressor learned from BioAmass
        # 使用从BioAmass学习的回归器回归解剖学关节位置
        self.register_buffer('J_regressor_osim', torch.FloatTensor(skel_data['J_regressor_osim']))   
        self.register_buffer('joint_sockets', torch.FloatTensor(skel_data['joint_sockets']))
        
        self.register_buffer('per_joint_rot', torch.FloatTensor(skel_data['per_joint_rot']))
        
        # Skin model skinning weights
        # 皮肤模型蒙皮权重
        self.register_buffer('weights', torch.FloatTensor(skel_data['weights']))

        # Skeleton model skinning weights
        # 骨骼模型蒙皮权重
        self.register_buffer('skel_weights', torch.FloatTensor(skel_data['skel_weights']))        
        self.register_buffer('skel_weights_rigid', torch.FloatTensor(skel_data['skel_weights_rigid']))        
        
        # Kinematic tree of the model
         # 模型的运动学树
        self.register_buffer('kintree_table', torch.from_numpy(skel_data['osim_kintree_table'].astype(np.int64)))
        # self.register_buffer('osim_kintree_table', torch.from_numpy(skel_data['osim_kintree_table'].astype(np.int64)))
        self.register_buffer('parameter_mapping', torch.from_numpy(skel_data['parameter_mapping'].astype(np.int64)))
        
        # transformation from osim can pose to T pose
        # 从osim标准姿势到T姿势的变换
        self.register_buffer('tpose_transfo', torch.FloatTensor(skel_data['tpose_transfo']))
        
        # transformation from osim can pose to A pose
        # 从osim标准姿势到A姿势的变换
        self.register_buffer('apose_transfo', torch.FloatTensor(skel_data['apose_transfo']))
        self.register_buffer('apose_rel_transfo', torch.FloatTensor(skel_data['apose_rel_transfo']))
        
        # Indices of bones which orientation should not vary with beta in T pose:
         # T姿势中方向不应随beta变化的骨骼索引：
        joint_idx_fixed_beta = [0, 5, 10, 13, 18, 23]
        self.register_buffer('joint_idx_fixed_beta', torch.LongTensor(joint_idx_fixed_beta))                   
        
        id_to_col = {self.kintree_table[1, i].item(): i for i in range(self.kintree_table.shape[1])}
        self.register_buffer('parent', torch.LongTensor(
            [id_to_col[self.kintree_table[0, it].item()] for it in range(1, self.kintree_table.shape[1])]))


        # child array
        # 子数组
        # TODO create this array in the SKEL creator
        child_array = []
        Nj = self.num_joints
        for i in range(0, Nj):
            try:
                j_array = torch.where(self.kintree_table[0] == i)[0] # candidate child lines 候选子行
                if len(j_array) == 0:
                    child_index = 0
                else:
                    
                    j = j_array[0]
                    if j>=len(self.kintree_table[1]):
                        child_index = 0
                    else:
                        child_index = self.kintree_table[1,j].item()
                child_array.append(child_index)
            except:
                import ipdb; ipdb.set_trace()

        # print(f"child_array: ")
        # [print(i,child_array[i]) for i in range(0, Nj)]
        self.register_buffer('child', torch.LongTensor(child_array))
        
        # Instantiate joints
        # 实例化关节
        # 1. 骨盆、股骨等大关节使用CustomJoint，允许三个自由度的旋转。
        # 2. 膝关节使用特殊的WalkerKnee类，考虑了膝关节特有的运动特性。
        # 3. 踝关节相关的骨骼（距骨、跟骨、趾骨）使用PinJoint，这是一种单轴旋转关节，parent_frame_ori参数定义了父坐标系的方向。
        # 4. 脊柱（腰椎、胸椎）和头部使用ConstantCurvatureJoint，可能模拟了脊柱的弯曲特性。
        # 5. 肩胛骨使用EllipsoidJoint，这可能是为了模拟肩关节的球窝结构。
        # 6. 上肢的其他关节（肱骨、尺骨、桡骨、手）使用CustomJoint，但具有不同的旋转轴设置。
        self.joints_dict = nn.ModuleList([ 
            CustomJoint(axis=[[0,0,1], [1,0,0], [0,1,0]], axis_flip=[1, 1, 1]), # 0 骨盆 (pelvis)
            CustomJoint(axis=[[0,0,1], [1,0,0], [0,1,0]], axis_flip=[1, 1, 1]), # 1 右股骨 (femur_r)
            WalkerKnee(), # 2 右胫骨 (tibia_r)
            PinJoint(parent_frame_ori = [0.175895, -0.105208, 0.0186622]), # 3 右距骨 (talus_r) 参数来自.osim文件中Joint->frames->PhysicalOffsetFrame->orientation
            PinJoint(parent_frame_ori = [-1.76818999, 0.906223, 1.8196000]), # 4 右跟骨 (calcn_r)
            PinJoint(parent_frame_ori = [-3.141589999, 0.6199010, 0]), # 5 右趾骨 (toes_r)
            CustomJoint(axis=[[0,0,1], [1,0,0], [0,1,0]], axis_flip=[1, -1, -1]), # 6 左股骨 (femur_l)
            WalkerKnee(), # 7 左胫骨 (tibia_l)
            PinJoint(parent_frame_ori = [0.175895, -0.105208, 0.0186622]), # 8 左距骨 (talus_l)
            PinJoint(parent_frame_ori = [1.768189999 ,-0.906223, 1.8196000]), # 9 左跟骨 (calcn_l)
            PinJoint(parent_frame_ori = [-3.141589999, -0.6199010, 0]), # 10 左趾骨 (toes_l)
            ConstantCurvatureJoint(axis=[[1,0,0], [0,0,1], [0,1,0]], axis_flip=[1, 1, 1]), # 11 腰椎 (lumbar)
            ConstantCurvatureJoint(axis=[[1,0,0], [0,0,1], [0,1,0]], axis_flip=[1, 1, 1]), # 12 胸椎 (thorax)
            ConstantCurvatureJoint(axis=[[1,0,0], [0,0,1], [0,1,0]], axis_flip=[1, 1, 1]), # 13 头部 (head)
            EllipsoidJoint(axis=[[0,1,0], [0,0,1], [1,0,0]], axis_flip=[1, -1, -1]), # 14 右肩胛骨 (scapula_r)
            CustomJoint(axis=[[1,0,0], [0,1,0], [0,0,1]], axis_flip=[1, 1, 1]), # 15 右肱骨 (humerus_r)
            CustomJoint(axis=[[0.0494, 0.0366, 0.99810825]], axis_flip=[[1]]), # 16 右尺骨 (ulna_r)
            CustomJoint(axis=[[-0.01716099, 0.99266564, -0.11966796]], axis_flip=[[1]]), # 17 右桡骨 (radius_r)
            CustomJoint(axis=[[1,0,0], [0,0,-1]], axis_flip=[1, 1]), # 18 右手 (hand_r)
            EllipsoidJoint(axis=[[0,1,0], [0,0,1], [1,0,0]], axis_flip=[1, 1, 1]), # 19 左肩胛骨 (scapula_l)
            CustomJoint(axis=[[1,0,0], [0,1,0], [0,0,1]], axis_flip=[1, 1, 1]), # 20 左肱骨 (humerus_l)
            CustomJoint(axis=[[-0.0494, -0.0366, 0.99810825]], axis_flip=[[1]]), # 21 左尺骨 (ulna_l)
            CustomJoint(axis=[[0.01716099, -0.99266564, -0.11966796]], axis_flip=[[1]]), # 22 左桡骨 (radius_l)
            CustomJoint(axis=[[-1,0,0], [0,0,-1]], axis_flip=[1, 1]), # 23 左手 (hand_l)
        ])

        
    def pose_params_to_rot(self, osim_poses):
        """ Transform the pose parameters to 3x3 rotation matrices
        Each parameter is mapped to a joint as described in joint_dict.
        The specific joint object is then used to compute the rotation matrix.
        将姿势参数转换为3x3旋转矩阵
        每个参数按照joint_dict中的描述映射到一个关节。
        然后使用特定的关节对象计算旋转矩阵。
        
        参数：
        osim_poses: 姿势参数张量
        
        返回：
        Rp: 旋转矩阵
        tp: 平移向量
        """
    
        B = osim_poses.shape[0]
        Nj = self.num_joints
        
        ident = torch.eye(3, dtype=osim_poses.dtype).to(osim_poses.device)
        Rp = ident.unsqueeze(0).unsqueeze(0).repeat(B, Nj,1,1)
        tp = torch.zeros(B, Nj, 3).to(osim_poses.device)
        start_index = 0
        for i in range(0, Nj):
            joint_object = self.joints_dict[i]
            end_index = start_index + joint_object.nb_dof
            Rp[:, i] = joint_object.q_to_rot(osim_poses[:, start_index:end_index])
            start_index = end_index  
        return Rp, tp
    
        
    def params_name_to_index(self, param_name):
        """
        将参数名称转换为索引
        
        参数：
        param_name: 参数名称
        
        返回：
        参数索引
        """
        assert param_name in pose_param_names
        param_index = pose_param_names.index(param_name)
        return param_index
        
        
    def forward(self, poses, betas, trans, poses_type='skel', skelmesh=True):      
        """
        SKEL模型的前向传播函数
        params
            poses : B x 46 tensor of pose parameters B x 46 张量,表示姿势参数
            betas : B x 10 tensor of shape parameters, same as SMPL  B x 10 张量,表示形状参数,与SMPL相同
            trans : B x 3 tensor of translation B x 3 张量,表示平移
            poses_type : str, 'skel', should not be changed 字符串,'skel',不应改变
            skelemesh : bool, if True, returns the skeleton vertices. The skeleton mesh is heavy so to fit on GPU memory, set to False when not needed. 布尔值,如果为True,返回骨骼顶点。骨骼网格较大,为了适应GPU内存,不需要时设为False。

        return SKELOutput class with the following fields:
            SKELOutput类,包含以下字段:
            betas: Optional[Tensor] = None # 形状参数,控制身体形状变化
            body_pose: Optional[Tensor] = None # 身体姿势参数,不包括全局旋转
            skin_verts: Optional[Tensor] = None  # 变形后的皮肤顶点位置
            skel_verts: Optional[Tensor] = None # 变形后的骨骼顶点位置
            joints: Optional[Tensor] = None # 变形后的关节位置
            joints_ori: Optional[Tensor] = None # 关节方向,通常表示为3x3旋转矩阵
            poses: Optional[Tensor] = None # 完整的姿势参数,包括全局旋转
            trans : Optional[Tensor] = None # 全局平移参数
            pose_offsets : Optional[Tensor] = None # 由姿势引起的顶点偏移量
        
        In this function we use the following conventions:
        在此函数中,我们使用以下约定:
        B : batch size 批次大小
        Ns : skin vertices 皮肤顶点数
        Nk : skeleton vertices 骨骼顶点数
        """
        
        # 获取皮肤和骨骼顶点数,关节数和批次大小
        Ns = self.skin_template_v.shape[0] # nb skin vertices
        Nk = self.skel_template_v.shape[0] # nb skeleton vertices
        Nj = self.num_joints
        B = poses.shape[0]
        device = poses.device
        
        # Check the shapes of the inputs
        # 检查输入的形状
        assert len(betas.shape) == 2, f"Betas should be of shape (B, {self.num_betas}), but got {betas.shape}"
        assert poses.shape[0] == betas.shape[0], f"Expected poses and betas to have the same batch size, but got {poses.shape[0]} and {betas.shape[0]}"
        assert poses.shape[0] == trans.shape[0], f"Expected poses and betas to have the same batch size, but got {poses.shape[0]} and {trans.shape[0]}"
        
        # Check the device of the inputs
        # 检查输入的设备
        assert betas.device == device, f"Betas should be on device {device}, but got {betas.device}"
        assert trans.device == device, f"Trans should be on device {device}, but got {trans.device}"  
        
        # 获取模板顶点
        skin_v0 = self.skin_template_v[None, :]
        skel_v0 = self.skel_template_v[None, :]
        betas = betas[:, :, None] # TODO Name the expanded beta differently
        
        # TODO
        # 处理不同类型的姿势参数
        assert poses_type in ['skel', 'bsm'], f"got {poses_type}"
        if poses_type == 'bsm':
            assert poses.shape[1] == self.num_q_params - 3, f'With poses_type bsm, expected parameters of shape (B, {self.num_q_params - 3}, got {poses.shape}'
            poses_bsm = poses
            poses_skel = torch.zeros(B, self.num_q_params)
            poses_skel[:,:3] = poses_bsm[:, :3]
            trans = poses_bsm[:, 3:6] # In BSM parametrization, the hips translation is given by params 3 to 5 在BSM参数化中,髋部平移由参数3到5给出
            poses_skel[:, 3:] = poses_bsm
            poses = poses_skel
   
        else:
            assert poses.shape[1] == self.num_q_params, f'With poses_type skel, expected parameters of shape (B, {self.num_q_params}), got {poses.shape}'
            pass
        # Load poses as expected
        # Distinction bsm skel. by default it will be bsm

        # ------- Shape 形状变形 ----------
        # Apply the beta offset to the template
        # 将beta偏移应用到模板
        shapedirs  = self.shapedirs.view(-1, self.num_betas)[None, :].expand(B, -1, -1) # B x D*Ns x num_betas
        v_shaped = skin_v0 + torch.matmul(shapedirs, betas).view(B, Ns, 3)
        
        # ------- Joints 关节回归 ----------
        # Regress the anatomical joint location
        # 回归解剖学关节位置
        J = torch.einsum('bik,ji->bjk', [v_shaped, self.J_regressor_osim]) # BxJx3 # osim regressor
        # J = self.apose_transfo[:, :3, -1].view(1, Nj, 3).expand(B, -1, -1)  # Osim default pose joints location
        
        
        # Local translation
        # 局部平移
        J_ = J.clone() # BxJx3
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        t = J_[:, :, :, None] # BxJx3x1
        
        # ------- Bones transformation matrix 骨骼变换矩阵 ----------
        
        # Bone initial transform to go from unposed to SMPL T pose
        # 从无姿态到SMPL T姿态的初始骨骼变换
        Rk01 = self.compute_bone_orientation(J, J_)
         
        # BSM default pose rotations
        # BSM 默认姿态旋转
        Ra = self.apose_rel_transfo[:, :3, :3].view(1, Nj, 3,3).expand(B, Nj, 3, 3) 

        # Local bone rotation given by the pose param
        # 由姿态参数给出的局部骨骼旋转
        Rp, tp = self.pose_params_to_rot(poses)  # BxNjx3x3 pose params to rotation
            
        R = matmul_chain([Rk01, Ra.transpose(2,3), Rp, Ra, Rk01.transpose(2,3)])

        ###### Compute translation for non pure rotation joints
        ###### 计算非纯旋转关节的平移  
        t_posed = t.clone()
        
        # Scapula
        # 肩胛骨
        thorax_width = torch.norm(J[:, 19, :] - J[:, 14, :], dim=1) # Distance between the two scapula joints, size B
        thorax_height = torch.norm(J[:, 12, :] - J[:, 11, :], dim=1) # Distance between the two scapula joints, size B
        
        angle_abduction = poses[:,26]
        angle_elevation = poses[:,27]
        angle_rot = poses[:,28]
        angle_zero = torch.zeros_like(angle_abduction)
        t_posed[:,14] = t_posed[:,14] + \
                        (right_scapula(angle_abduction, angle_elevation, angle_rot, thorax_width, thorax_height).view(-1,3,1)
                          - right_scapula(angle_zero, angle_zero, angle_zero, thorax_width, thorax_height).view(-1,3,1))


        angle_abduction = poses[:,36]
        angle_elevation = poses[:,37]
        angle_rot = poses[:,38]
        angle_zero = torch.zeros_like(angle_abduction)
        t_posed[:,19] = t_posed[:,19] + \
                        (left_scapula(angle_abduction, angle_elevation, angle_rot, thorax_width, thorax_height).view(-1,3,1) 
                         - left_scapula(angle_zero, angle_zero, angle_zero, thorax_width, thorax_height).view(-1,3,1))
               
               
        # Knee_r
        # TODO add the Walker knee offset
        # bone_scale = self.compute_bone_scale(J_,J, skin_v0, v_shaped)
        # f1 = poses[:, 2*3+2].clone()
        # scale_femur = bone_scale[:, 2]
        # factor = 0.076/0.080 * scale_femur # The template femur medial laterak spacing #66
        # f = -f1*180/torch.pi #knee_flexion
        # varus = (0.12367*f)-0.0009*f**2
        # introt = 0.3781*f-0.001781*f**2
        # ydis = (-0.0683*f 
        #         + 8.804e-4 * f**2 
        #         - 3.750e-06*f**3
        #         )/1000*factor # up-down
        # zdis = (-0.1283*f 
        #         + 4.796e-4 * f**2)/1000*factor # 
        # import ipdb; ipdb.set_trace()
        # poses[:, 9] = poses[:, 9] + varus
        # t_posed[:,2] = t_posed[:,2] + torch.stack([torch.zeros_like(ydis), ydis, zdis], dim=1).view(-1,3,1)
        # poses[:, 2*3+2]=0
        
        # t_unposed = torch.zeros_like(t_posed)
        # t_unposed[:,2] = torch.stack([torch.zeros_like(ydis), ydis, zdis], dim=1).view(-1,3,1)
        
                        
        # Spine
        # 脊椎 
        lumbar_bending = poses[:,17]
        lumbar_extension = poses[:,18]
        angle_zero = torch.zeros_like(lumbar_bending)
        interp_t = torch.ones_like(lumbar_bending)
        l = torch.abs(J[:, 11, 1] - J[:, 0, 1]) # Length of the spine section along y axis 沿y轴的脊柱段长度
        t_posed[:,11] = t_posed[:,11] + \
                        (curve_torch_3d(lumbar_bending, lumbar_extension, t=interp_t, l=l)
                         - curve_torch_3d(angle_zero, angle_zero, t=interp_t, l=l))
 
        thorax_bending = poses[:,20]
        thorax_extension = poses[:,21]
        angle_zero = torch.zeros_like(thorax_bending)
        interp_t = torch.ones_like(thorax_bending)
        l = torch.abs(J[:, 12, 1] - J[:, 11, 1]) # Length of the spine section 脊柱段长度

        t_posed[:,12] = t_posed[:,12] + \
                        (curve_torch_3d(thorax_bending, thorax_extension, t=interp_t, l=l)
                         - curve_torch_3d(angle_zero, angle_zero, t=interp_t, l=l))                                               

        head_bending = poses[:, 23]
        head_extension = poses[:,24]
        angle_zero = torch.zeros_like(head_bending)
        interp_t = torch.ones_like(head_bending)
        l = torch.abs(J[:, 13, 1] - J[:, 12, 1]) # Length of the spine section 脊柱段长度
        t_posed[:,13] = t_posed[:,13] + \
                        (curve_torch_3d(head_bending, head_extension, t=interp_t, l=l)
                         - curve_torch_3d(angle_zero, angle_zero, t=interp_t, l=l)) 
                                                    
  
        # ------- Body surface transformation matrix 身体表面变换矩阵 ----------           
                                
        G_ = torch.cat([R, t_posed], dim=-1) # BxJx3x4 local transformation matrix 局部变换矩阵
        pad_row = torch.FloatTensor([0, 0, 0, 1]).to(device).view(1, 1, 1, 4).expand(B, Nj, -1, -1) # BxJx1x4
        G_ = torch.cat([G_, pad_row], dim=2) # BxJx4x4 padded to be 4x4 matrix an enable multiplication for the kinematic chain 填充为4x4矩阵以启用运动链乘法
        
        # Global transform
        # 全局变换
        G = [G_[:, 0].clone()]
        for i in range(1, Nj):
            G.append(torch.matmul(G[self.parent[i - 1]], G_[:, i, :, :]))
        G = torch.stack(G, dim=1)
        
        # ------- Pose dependant blend shapes 姿态相关混合形状 ----------
        # Note : Those should be retrained for SKEL as the SKEL joints location are different from SMPL.
        # 注意:这些应该为SKEL重新训练,因为SKEL关节位置与SMPL不同。
        # But the current version lets use get decent pose dependant deformations for the shoulders, belly and knies
        # 但当前版本让我们为肩膀、腹部和膝盖获得不错的姿态相关变形
        ident = torch.eye(3, dtype=v_shaped.dtype, device=device)
        
        # We need the per SMPL joint bone transform to compute pose dependant blend shapes.
        # 我们需要每个SMPL关节的骨骼变换来计算姿态相关混合形状
        # Initialize each joint rotation with identity
        # 用单位矩阵初始化每个关节旋转
        Rsmpl = ident.unsqueeze(0).unsqueeze(0).expand(B, self.num_joints_smpl, -1, -1) # BxNjx3x3 
        
        Rskin = G_[:, :, :3, :3] # BxNjx3x3
        Rsmpl[:, smpl_joint_corresp] = Rskin.clone()[:] # BxNjx3x3 pose params to rotation 姿态参数到旋转
        pose_feature = Rsmpl[:, 1:].view(B, -1, 3, 3) - ident
        pose_offsets = torch.matmul(pose_feature.view(B, -1),
                                    self.posedirs.view(Ns*3, -1).T).view(B, -1, 3)
        v_shaped_pd = v_shaped + pose_offsets
          
        
        ##########################################################################################
        #Transform skin mesh 变换皮肤网格
        ############################################################################################

        # Apply global transformation to the template mesh
        # 将全局变换应用到模板网格
        rest = torch.cat([J, torch.zeros(B, Nj, 1).to(device)], dim=2).view(B, Nj, 4, 1) # BxJx4x1
        zeros = torch.zeros(B, Nj, 4, 3).to(device) # BxJx4x3
        rest = torch.cat([zeros, rest], dim=-1) # BxJx4x4
        rest = torch.matmul(G, rest) # This is a 4x4 transformation matrix that only contains translation to the rest pose joint location 这是一个4x4变换矩阵,仅包含到静止姿态关节位置的平移
        Gskin = G - rest
        
        # Compute per vertex transformation matrix (after weighting)
        # 计算每个顶点的变换矩阵(加权后)
        T = torch.matmul(self.weights, Gskin.permute(1, 0, 2, 3).contiguous().view(Nj, -1)).view(Ns, B, 4,4).transpose(0, 1)
        rest_shape_h = torch.cat([v_shaped_pd, torch.ones_like(v_shaped_pd)[:, :, [0]]], dim=-1)
        v_posed = torch.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]
        
        # translation
        # 平移
        v_trans = v_posed + trans[:,None,:]
        
        ##########################################################################################
        #Transform joints 变换关节
        ############################################################################################

        # import ipdb; ipdb.set_trace()
        root_transform = with_zeros(torch.cat((R[:,0],J[:,0][:,:,None]),2))
        results =  [root_transform]
        for i in range(0, self.parent.shape[0]):
            transform_i =  with_zeros(torch.cat((R[:, i + 1], t_posed[:,i+1]), 2))
            curr_res = torch.matmul(results[self.parent[i]],transform_i)
            results.append(curr_res)
        results = torch.stack(results, dim=1)
        posed_joints = results[:, :, :3, 3]
        J_transformed = posed_joints + trans[:,None,:]
        
    
        ##########################################################################################
        # Transform skeleton 变换骨骼
        ############################################################################################

        if skelmesh:
            G_bones = None
            # Shape the skeleton by scaling its bones
            # 通过缩放其骨骼来塑造骨架
            skel_rest_shape_h = torch.cat([skel_v0, torch.ones_like(skel_v0)[:, :, [0]]], dim=-1).expand(B, Nk, -1) # (1,Nk,3)

            # compute the bones scaling from the kinematic tree and skin mesh
             # 从运动学树和皮肤网格计算骨骼缩放
            bone_scale = self.compute_bone_scale(J_, v_shaped, skin_v0)
                        
            # Apply bone meshes scaling:
            # 应用骨骼网格缩放:
            skel_v_shaped = torch.cat([(torch.matmul(bone_scale[:,:,0], self.skel_weights_rigid.T) * skel_rest_shape_h[:, :, 0])[:, :, None], 
                                    (torch.matmul(bone_scale[:,:,1], self.skel_weights_rigid.T) * skel_rest_shape_h[:, :, 1])[:, :, None],
                                    (torch.matmul(bone_scale[:,:,2], self.skel_weights_rigid.T) * skel_rest_shape_h[:, :, 2])[:, :, None],
                                    (torch.ones(B, Nk, 1).to(device))
                                    ], dim=-1) 
                
            
            # Align the bones with the proper axis
            # 将骨骼与适当的轴对齐
            Gk01 = build_homog_matrix(Rk01, J.unsqueeze(-1)) # BxJx4x4
            T = torch.matmul(self.skel_weights_rigid, Gk01.permute(1, 0, 2, 3).contiguous().view(Nj, -1)).view(Nk, B, 4,4).transpose(0, 1) #[1, 48757, 3, 3]
            skel_v_align = torch.matmul(T, skel_v_shaped[:, :, :, None])[:, :, :, 0]
            
            # This transfo will be applied with weights, effectively unposing the whole skeleton mesh in each joint frame. 
            # 这个变换将通过权重应用,有效地将整个骨骼网格在每个关节框架中解除姿态。
            # Then, per joint weighted transformation can then be applied
            # 然后,可以应用每个关节的加权变换
            G_tpose_to_unposed = build_homog_matrix(torch.eye(3).view(1,1,3,3).expand(B, Nj, 3, 3).to(device), -J.unsqueeze(-1)) # BxJx4x4
            G_skel = torch.matmul(G, G_tpose_to_unposed)            
            G_bones = torch.matmul(G, Gk01)

            T = torch.matmul(self.skel_weights, G_skel.permute(1, 0, 2, 3).contiguous().view(Nj, -1)).view(Nk, B, 4,4).transpose(0, 1)
            skel_v_posed = torch.matmul(T, skel_v_align[:, :, :, None])[:, :, :3, 0]
            
            skel_trans = skel_v_posed + trans[:,None,:]

        else:
            skel_trans = skel_v0
            Gk01 = build_homog_matrix(Rk01, J.unsqueeze(-1)) # BxJx4x4
            G_bones = torch.matmul(G, Gk01)

        # 准备输出
        joints = J_transformed
        skin_verts = v_trans
        skel_verts = skel_trans       
        joints_ori = G_bones[:,:,:3,:3]
        
        if skin_verts.max() > 1e3:
            import ipdb; ipdb.set_trace()
        
        # 创建并返回输出对象
        output = SKELOutput(skin_verts=skin_verts,
                            skel_verts=skel_verts,
                            joints=joints,
                            joints_ori=joints_ori,
                            betas=betas,
                            poses=poses,
                            trans = trans,
                            pose_offsets = pose_offsets)

        return output

    
    def compute_bone_scale(self, J_, v_shaped, skin_v0):
        """
        计算骨骼的缩放比例
        
        参数:
        J_: 关节位置差分 (B, Nj, 3)
        v_shaped: 形状变形后的顶点 (B, Ns, 3)
        skin_v0: 皮肤模板顶点 (1, Ns, 3)
        
        返回:
        bone_scale: 每个骨骼的缩放比例 (B, Nj, 3)
        """
        # index                         [0,  1,     2,     3      4,     5,   , ...] # TODO add last one, figure out bone scale indices
        # J_ bone vectors               [j0, j1-j0, j2-j0, j3-j0, j4-j1, j5-j2, ...]
        # norm(J) = length of the bone  [j0, j1-j0, j2-j0, j3-j0, j4-j1, j5-j2, ...]
        # self.joints_sockets           [j0, j1-j0, j2-j0, j3-j0, j4-j1, j5-j2, ...]
        # self.skel_weights             [j0, j1,    j2,    j3,    j4,    j5,    ...]
        B = J_.shape[0]
        Nj = J_.shape[1]
        
        # 初始化骨骼缩放为1
        bone_scale = torch.ones(B, Nj).to(J_.device)
        
        # BSM template joints location
        # BSM模板关节位置
        osim_joints_r = self.apose_rel_transfo[:, :3, 3].view(1, Nj, 3).expand(B, Nj, 3).clone()
        
         # 计算BSM模板和SMPL模型的骨骼长度
        length_bones_bsm = torch.norm(osim_joints_r, dim=-1).expand(B, -1)
        length_bones_smpl = torch.norm(J_, dim=-1) # (B, Nj)
        # 计算父骨骼的缩放比例
        bone_scale_parent = length_bones_smpl / length_bones_bsm
        
        # 对非叶子节点应用父骨骼的缩放
        non_leaf_node = (self.child != 0)
        bone_scale[:,non_leaf_node] = (bone_scale_parent[:,self.child])[:,non_leaf_node]

        # Ulna should have the same scale as radius
        # 尺骨应与桡骨具有相同的缩放
        bone_scale[:,16] = bone_scale[:,17]
        bone_scale[:,16] = bone_scale[:,17]

        bone_scale[:,21] = bone_scale[:,22]
        bone_scale[:,21] = bone_scale[:,22]  
        
        # Thorax
        # Thorax scale is defined by the relative position of the thorax to its child joint, not parent joint as for other bones
        # 胸部缩放由其子关节的相对位置定义,而不是父关节
        bone_scale[:, 12] = bone_scale[:, 11] 
        
        # Lumbars 
        # Lumbar scale is defined by the y relative position of the lumbar joint
        # 腰部缩放由腰部关节的y相对位置定义
        length_bones_bsm = torch.abs(osim_joints_r[:,11, 1])
        length_bones_smpl = torch.abs(J_[:, 11, 1]) # (B, Nj)
        bone_scale_lumbar = length_bones_smpl / length_bones_bsm
        bone_scale[:, 11] = bone_scale_lumbar
        
        # Expand to 3 dimensions and adjest scaling to avoid skin-skeleton intersection and handle the scaling of leaf body parts (hands, feet)
        # 扩展到3个维度,调整缩放以避免皮肤-骨骼交叉并处理叶子身体部位(手、脚)的缩放
        bone_scale = bone_scale.reshape(B, Nj, 1).expand(B, Nj, 3).clone()
            
        for (ji, doi, dsi), (v1, v2) in scaling_keypoints.items():
            bone_scale[:, ji, doi] = ((v_shaped[:,v1] - v_shaped[:, v2])/ (skin_v0[:,v1] - skin_v0[:, v2]))[:,dsi] # Top over chin       
            #TODO: Add keypoints for feet scaling in scaling_keypoints
        
        # Adjust thorax front-back scaling
        # 调整胸部前后缩放
        # TODO fix this part
        v1 = 3027 #thorax back 胸部后
        v2 = 3495 #thorax front 胸部前
        
        scale_thorax_up = ((v_shaped[:,v1] - v_shaped[:, v2])/ (skin_v0[:,v1] - skin_v0[:, v2]))[:,2]  # good for large people 适用于大体型人
        v2 = 3506 #sternum 胸骨
        scale_thorax_sternum = ((v_shaped[:,v1] - v_shaped[:, v2])/ (skin_v0[:,v1] - skin_v0[:, v2]))[:,2] # Good for skinny people 适用于瘦小体型人
        bone_scale[:, 12, 0] = torch.min(scale_thorax_up, scale_thorax_sternum) # Avoids super expanded ribcage for large people and sternum outside for skinny people 避免大体型人的胸腔过度扩张和瘦小体型人的胸骨外露
                        
        #lumbars, adjust width to be same as thorax 腰部,调整宽度与胸部相同
        bone_scale[:, 11, 0] = bone_scale[:, 12, 0]
        
        return bone_scale
        
   
    
    def compute_bone_orientation(self, J, J_):        
        """Compute each bone orientation in T pose
        计算T姿势中每个骨骼的方向

        参数:
        J: 关节位置 (B, Nj, 3)
        J_: 关节位置差分 (B, Nj, 3)
        
        返回:
        Gk: 每个骨骼的旋转矩阵 (B, Nj, 3, 3)
        """
        
        # method = 'unposed'
        # method = 'learned'
        method = 'learn_adjust'
        
        B = J_.shape[0]
        Nj = J_.shape[1]

        # Create an array of bone vectors the bone meshes should be aligned to.
        # 创建骨骼向量数组,骨骼网格应与之对齐
        bone_vect = torch.zeros_like(J_) # / torch.norm(J_, dim=-1)[:, :, None] # (B, Nj, 3)
        bone_vect[:] = J_[:, self.child] # Most bones are aligned between their parent and child joint 大多数骨骼在其父关节和子关节之间对齐
        bone_vect[:,16] = bone_vect[:,16]+bone_vect[:,17] # We want to align the ulna to the segment joint 16 to 18 我们希望尺骨与关节16到18的段对齐
        bone_vect[:,21] = bone_vect[:,21]+bone_vect[:,22] # Same other ulna 另一侧尺骨同理
        
        # TODO Check indices here
        # bone_vect[:,13] = bone_vect[:,12].clone() 
        bone_vect[:,12] = bone_vect.clone()[:,11].clone() # We want to align the  thorax on the thorax-lumbar segment 我们希望胸部与胸部-腰部段对齐
        # bone_vect[:,11] = bone_vect[:,0].clone() 
        
        osim_vect = self.apose_rel_transfo[:, :3, 3].clone().view(1, Nj, 3).expand(B, Nj, 3).clone()
        osim_vect[:] = osim_vect[:,self.child]
        osim_vect[:,16] = osim_vect[:,16]+osim_vect[:,17] # We want to align the ulna to the segment joint 16 to 18 我们希望尺骨与关节16到18的段对齐
        osim_vect[:,21] = osim_vect[:,21]+osim_vect[:,22] # We want to align the ulna to the segment joint 16 to 18 另一侧尺骨同理
        
        # TODO: remove when this has been checked 
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(osim_vect[:,0,0], osim_vect[:,0,1], osim_vect[:,0,2], color='r')
        # plt.show()
        
        Gk = torch.eye(3, device=J_.device).repeat(B, Nj, 1, 1)  
        
        if method == 'unposed':
            return Gk

        elif method == 'learn_adjust':
            Gk_learned = self.per_joint_rot.view(1, Nj, 3, 3).expand(B, -1, -1, -1) #load learned rotation 加载学习的旋转
            osim_vect_corr = torch.matmul(Gk_learned, osim_vect.unsqueeze(-1)).squeeze(-1)
                
            Gk[:,:] = rotation_matrix_from_vectors(osim_vect_corr, bone_vect)
            # set nan to zero 将nan设为零
            # TODO: Check again why the following line was required
            Gk[torch.isnan(Gk)] = 0
            
            # 某些关节的方向不应随beta变化,保持不变
            # Gk[:,[18,23]] = Gk[:,[16,21]] # hand has same orientation as ulna
            # Gk[:,[5,10]] = Gk[:,[4,9]] # toe has same orientation as calcaneus
            # Gk[:,[0,11,12,13,14,19]] = torch.eye(3, device=J_.device).view(1,3,3).expand(B, 6, 3, 3) # pelvis, torso and shoulder blade orientation does not vary with beta, leave it
            Gk[:, self.joint_idx_fixed_beta] =  torch.eye(3, device=J_.device).view(1,3,3).expand(B, len(self.joint_idx_fixed_beta), 3, 3) # pelvis, torso and shoulder blade orientation should not vary with beta, leave it
               
            Gk = torch.matmul(Gk, Gk_learned)

        elif method == 'learned':
            """ Apply learned transformation 应用学习的变换 """       
            Gk = self.per_joint_rot.view(1, Nj, 3, 3).expand(B, -1, -1, -1)
            
        else:
            raise NotImplementedError
        
        return Gk
