import torch
import numpy as np

def right_scapula(angle_abduction, angle_elevation, angle_rot, thorax_width, thorax_height):
    """
    计算右侧肩胛骨的位置
    
    参数:
    angle_abduction: 外展角度
    angle_elevation: 提升角度
    angle_rot: 旋转角度 (此函数中未使用)
    thorax_width: 胸廓宽度
    thorax_height: 胸廓高度
    
    返回:
    t: 右侧肩胛骨的3D位置坐标 (张量)
    """
    # 计算x、y、z方向的半径
    radius_x = thorax_width / 4 * torch.cos(angle_elevation-np.pi/4)
    radius_y = thorax_width / 4
    radius_z = thorax_height / 2
    # 计算肩胛骨的位置
    t = torch.stack([-radius_x * torch.cos(angle_abduction), 
                        -radius_z*torch.sin(angle_elevation-np.pi/4) ,  # Todo revert sin and cos here
                        radius_y * torch.sin(angle_abduction)
                    ], dim=1)
    
    # h = thorax_height
    # w = thorax_width
    # d = thorax_width/2 # approximation
    
    # theta1 = angle_abduction
    # theta2 = angle_elevation
    
    # tx = h*torch.sin(theta2) + h
    # ty = 0*tx
    # tz = 0*tx
    
    # t = torch.stack([tx, 
    #                  ty,  # Todo revert sin and cos here
    #                  tz
    #                 ], dim=1)
    return t


def left_scapula(angle_abduction, angle_elevation, angle_rot, thorax_width, thorax_height):
    """
    计算左侧肩胛骨的位置
    
    参数:
    angle_abduction: 外展角度
    angle_elevation: 提升角度
    angle_rot: 旋转角度 (此函数中未使用)
    thorax_width: 胸廓宽度
    thorax_height: 胸廓高度
    
    返回:
    t: 左侧肩胛骨的3D位置坐标 (张量)
    """
    # 对于左侧，角度需要取反
    angle_abduction = -angle_abduction
    angle_elevation = -angle_elevation
    
    # 计算x、y、z方向的半径
    radius_x = thorax_width / 4 * torch.cos(angle_elevation-np.pi/4)
    radius_y = thorax_width / 4
    radius_z = thorax_height / 2
    # 计算肩胛骨的位置
    t = torch.stack([radius_x * torch.cos(angle_abduction), 
                        -radius_z*torch.sin(angle_elevation-np.pi/4) , 
                        radius_y * torch.sin(angle_abduction)
                    ], dim=1)
    return t


def curve_torch_1d(angle, t, l):
    """Trace a curve with constan curvature of arc length l and curvature k using plt
    计算具有恒定曲率的曲线
    :param angle: angle of the curve in radians B 曲线的角度（弧度），形状为B
    :param t: parameter of the curve, float in [0,1] of shape B 曲线的参数，范围[0,1]，形状为B
    :param l: arc length of the curve, float of shape B 曲线的弧长，形状为B

    返回:
    x, y: 曲线上的点的x和y坐标
    """
    
    assert angle.shape == t.shape == l.shape, f"Shapes of angle, t and l must be the same, got {angle.shape}, {t.shape}, {l.shape}"
    
    # import ipdb; ipdb.set_trace()
    x = torch.zeros_like(angle)
    y = torch.zeros_like(angle)
    
    # 处理小角度的情况
    mask_small = torch.abs(angle) < 1e-5
    
    # Process small angles separately
    # We use taylor development for small angles to avoid explosion due to number precision
    tm = t[mask_small]
    anglem = angle[mask_small]
    lm = l[mask_small]
    # import ipdb; ipdb.set_trace()

     # 使用泰勒展开处理小角度
    x[mask_small] = lm * tm*tm* anglem/2
    y[mask_small] = lm * tm * (1 - tm*tm*tm/6 * anglem*anglem) 
    
    # Process non small angles
    # 处理非小角度的情况
    mask_big = torch.logical_not(mask_small)
    tm = t[mask_big]
    anglem = angle[mask_big]
    lm = l[mask_big]
    
    # print(x,y)  
    # return x,y
    c_arc = lm
    r = c_arc / (anglem)
    x[mask_big] = r*(1 - torch.cos(tm*anglem))
    y[mask_big] = r*torch.sin(tm*anglem)   
    return x,y

def curve_torch_3d(angle_x, angle_y, t, l):
    """
    计算3D空间中的曲线
    
    参数:
    angle_x: x方向的角度
    angle_y: y方向的角度
    t: 曲线的参数
    l: 曲线的弧长
    
    返回:
    3D空间中的曲线点坐标
    """
    # 计算x-y平面上的曲线
    x,y = curve_torch_1d(angle_x, t, l)
    tx = torch.cat([-x.unsqueeze(-1), 
                    y.unsqueeze(-1), 
                    torch.zeros_like(x).unsqueeze(-1)],
                   dim=1).unsqueeze(-1) # Extention 延伸
    # import ipdb; ipdb.set_trace()
    # 计算y-z平面上的曲线
    x,y = curve_torch_1d(angle_y, t, l)
    ty = torch.cat([torch.zeros_like(x).unsqueeze(-1), 
                    y.unsqueeze(-1), 
                    -x.unsqueeze(-1)],
                   dim=1).unsqueeze(-1) # Bending 弯曲
    # 组合两个平面上的曲线得到3D曲线
    return tx+ty
