import torch

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
    spine_indices = range(17, 25)
    
    spine_poses = poses[:, spine_indices]
    spine_loss = torch.linalg.norm(spine_poses, ord=2)
    return spine_loss

def compute_pose_loss(poses, pose_init):
    """
    计算整体姿势的损失（不包括全局旋转）
    
    参数:
    poses: 姿势参数张量
    
    返回:
    pose_loss: 整体姿势的L2范数损失
    """
    pose_loss = torch.linalg.norm(poses[:, 3:], ord=2) # The global rotation should not be constrained
    return pose_loss

def compute_anchor_pose(poses, pose_init):
    
    pose_loss = torch.nn.functional.mse_loss(poses[:, :3], pose_init[:, :3])
    return pose_loss 

def compute_anchor_trans(trans, trans_init):

    trans_loss = torch.nn.functional.mse_loss(trans, trans_init)
    return trans_loss 

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

def pretty_loss_print(loss_dict):
    # Pretty print the loss on the form loss val | loss1 val1 | loss2 val2 
    # Start with the total loss
    loss = sum(loss_dict.values())
    pretty_loss = f'{loss:.4f}'
    for key, val in loss_dict.items():
        pretty_loss += f' | {key} {val:.4f}'
    return pretty_loss   
