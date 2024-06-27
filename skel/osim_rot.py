import torch
from skel.joints_def import curve_torch_3d
from skel.utils import axis_angle_to_matrix, euler_angles_to_matrix, rodrigues

class OsimJoint(torch.nn.Module):
    """
    OpenSim关节的基类
    """
    def __init__(self) -> None:
        super().__init__()
        pass
    
    def q_to_translation(self, q, **kwargs):
        """
        将关节角度转换为平移
        
        参数:
        q: 关节角度
        
        返回:
        平移向量 (batch_size, 3)
        """              
        return torch.zeros(q.shape[0], 3).to(q.device)


class CustomJoint(OsimJoint):
    """
    自定义关节类
    """
    def __init__(self, axis, axis_flip) -> None:
        """
        参数:
        axis: 关节轴
        axis_flip: 轴反转标志
        """
        super().__init__()
        self.register_buffer('axis', torch.FloatTensor(axis))
        self.register_buffer('axis_flip', torch.FloatTensor(axis_flip))
        self.register_buffer('nb_dof', torch.tensor(len(axis)))  
        
    def q_to_rot(self, q, **kwargs):
        """
        将关节角度转换为旋转矩阵
        
        参数:
        q: 关节角度
        
        返回:
        旋转矩阵 (batch_size, 3, 3)
        """
        ident = torch.eye(3, dtype=q.dtype).to(q.device)
        
        Rp = ident.unsqueeze(0).expand(q.shape[0],3,3) # torch.eye(q.shape[0], 3, 3)
        for i in range(self.nb_dof):
            axis = self.axis[i]
            angle_axis = q[:, i:i+1] * self.axis_flip[i] * axis
            Rp_i = axis_angle_to_matrix(angle_axis)  
            Rp = torch.matmul(Rp_i, Rp)
        return Rp 
 

       
class CustomJoint1D(OsimJoint):
    """
    一维自定义关节类
    """
    def __init__(self, axis, axis_flip) -> None:
        """
        参数:
        axis: 关节轴
        axis_flip: 轴反转标志
        """
        super().__init__()
        self.axis = torch.FloatTensor(axis)
        self.axis = self.axis / torch.linalg.norm(self.axis)
        self.axis_flip = torch.FloatTensor(axis_flip)
        self.nb_dof = 1
        
    def q_to_rot(self, q, **kwargs):
        """
        将关节角度转换为旋转矩阵
        
        参数:
        q: 关节角度
        
        返回:
        旋转矩阵 (batch_size, 3, 3)
        """
        axis = self.axis
        angle_axis = q[:, 0:1] * self.axis_flip * axis  # 计算角轴
        Rp_i = axis_angle_to_matrix(angle_axis) # 将角轴转换为旋转矩阵
        return Rp_i    

    
class WalkerKnee(OsimJoint):
    """
    步行者膝关节类
    """
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer('nb_dof', torch.tensor(1))
        # self.nb_dof = 1
        
    def q_to_rot(self, q, **kwargs):
        """
        将膝关节角度转换为旋转矩阵
        
        参数:
        q: 膝关节角度
        
        返回:
        旋转矩阵 (batch_size, 3, 3)
        """
        # TODO : for now implement a basic knee
        theta_i = torch.zeros(q.shape[0], 3).to(q.device)
        theta_i[:, 2] = -q[:, 0] # 绕z轴旋转
        Rp_i = axis_angle_to_matrix(theta_i) 
        return Rp_i
        
class PinJoint(OsimJoint):
    """
    销关节类，用于踝关节
    """
    def __init__(self, parent_frame_ori) -> None:
        """
        参数:
        parent_frame_ori: 父坐标系方向
        """
        super().__init__()
        self.register_buffer('parent_frame_ori', torch.FloatTensor(parent_frame_ori))
        self.register_buffer('nb_dof', torch.tensor(1))
        
     
    def q_to_rot(self, q, **kwargs):
        """
        将销关节角度转换为旋转矩阵
        
        参数:
        q: 销关节角度
        
        返回:
        旋转矩阵 (batch_size, 3, 3)
        """
        
        talus_orient_torch = self.parent_frame_ori 
        Ra_i = euler_angles_to_matrix(talus_orient_torch, 'XYZ') # 跟骨方向
        
        z_axis = torch.FloatTensor([0,0,1]).to(q.device)
        axis = torch.matmul(Ra_i, z_axis).to(q.device)  # 计算旋转轴
        
        axis_angle = q[:, 0:1] * axis # 计算角轴
        Rp_i = axis_angle_to_matrix(axis_angle) # 将角轴转换为旋转矩阵
                
        return Rp_i
    
    
class ConstantCurvatureJoint(CustomJoint):
    """
    恒定曲率关节类，用于脊柱等
    """
    def __init__(self, **kwargs ) -> None:
        super().__init__( **kwargs)
        

        
class EllipsoidJoint(CustomJoint):
    """
    椭球关节类，用于肩关节等
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
