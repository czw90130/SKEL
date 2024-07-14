import math
"""
这段代码定义了一个人体骨骼模型的各种参数和限制。主要包括以下几个部分：
1. 骨骼关节名称列表(`skel_joints_name`):定义了人体骨骼的24个关节点。
2. 姿态参数名称列表(`pose_param_names`):定义了46个姿态参数,包括各个关节的旋转和角度。
3. 姿态限制范围字典(`pose_limits`):定义了部分姿态参数的运动范围限制。
4. SKEL骨骼与SMPL模型关节的对应关系(`smpl_joint_corresp`):建立了自定义骨骼模型与SMPL模型之间的关节对应关系。
5. 骨骼缩放关键点(`scaling_keypoints`):定义了用于骨骼缩放的关键顶点。
"""

# 骨骼关节名称列表
skel_joints_name= [
'pelvis', #0 骨盆
'femur_r', #1 右股骨
'tibia_r', #2 右胫骨
'talus_r', #3 右距骨
'calcn_r', #4 右跟骨
'toes_r', #5 右脚趾
'femur_l', #6 左股骨  
'tibia_l', #7 左胫骨
'talus_l', #8 左距骨
'calcn_l', #9 左跟骨
'toes_l', #10 左脚趾
'lumbar_body', #11 腰椎体
'thorax', #12 胸廓
'head', #13 头部
'scapula_r', #14 右肩胛骨
'humerus_r', #15 右肱骨
'ulna_r', #16 右尺骨
'radius_r', #17 右桡骨
'hand_r', #18 右手
'scapula_l', #19 左肩胛骨
'humerus_l', #20 左肱骨
'ulna_l', #21 左尺骨
'radius_l', #22 左桡骨
'hand_l'] #23 左手

# 姿态参数名称列表
pose_param_names = [
 'pelvis_tilt', #0 骨盆前倾/后倾 
 'pelvis_list', #1 骨盆侧倾
 'pelvis_rotation', #2 骨盆旋转
 'hip_flexion_r', #3 右髋关节屈曲
 'hip_adduction_r', #4 右髋关节内收
 'hip_rotation_r', #5 右髋关节旋转
 'knee_angle_r', #6 右膝关节角度
 'ankle_angle_r', #7 右踝关节角度
 'subtalar_angle_r', #8 右距下关节角度
 'mtp_angle_r', #9 右跖趾关节角度
 'hip_flexion_l', #10 左髋关节屈曲
 'hip_adduction_l', #11 左髋关节内收
 'hip_rotation_l', #12 左髋关节旋转
 'knee_angle_l', #13 左膝关节角度
 'ankle_angle_l', #14 左踝关节角度
 'subtalar_angle_l', #15 左距下关节角度
 'mtp_angle_l', #16 左跖趾关节角度
 'lumbar_bending', #17 腰椎侧屈
 'lumbar_extension', #18 腰椎伸展
 'lumbar_twist', #19 腰椎扭转
 'thorax_bending', #20 胸廓侧屈
 'thorax_extension', #21 胸廓伸展
 'thorax_twist', #22 胸廓扭转
 'head_bending', #23 头部侧屈
 'head_extension', #24 头部伸展
 'head_twist', #25 头部旋转
 'scapula_abduction_r', #26 右肩胛骨外展
 'scapula_elevation_r', #27 右肩胛骨上提
 'scapula_upward_rot_r', #28 右肩胛骨上旋
 'shoulder_r_x', #29 右肩关节X轴旋转
 'shoulder_r_y', #30 右肩关节Y轴旋转
 'shoulder_r_z', #31 右肩关节Z轴旋转
 'elbow_flexion_r', #32 右肘关节屈曲
 'pro_sup_r', #33 右前臂旋前/旋后
 'wrist_flexion_r', #34 右腕关节屈曲
 'wrist_deviation_r', #35 右腕关节偏移
 'scapula_abduction_l', #36 左肩胛骨外展
 'scapula_elevation_l', #37 左肩胛骨上提
 'scapula_upward_rot_l', #38 左肩胛骨上旋
 'shoulder_l_x', #39 左肩关节X轴旋转
 'shoulder_l_y', #40 左肩关节Y轴旋转
 'shoulder_l_z', #41 左肩关节Z轴旋转
 'elbow_flexion_l', #42 左肘关节屈曲
 'pro_sup_l', #43 左前臂旋前/旋后
 'wrist_flexion_l', #44 左腕关节屈曲
 'wrist_deviation_l', #45 左腕关节偏移
]

# 姿态限制范围字典
pose_limits = {
'scapula_abduction_r' :  [-0.628, 0.628], # 右肩胛骨外展范围
'scapula_elevation_r' :  [-0.4, -0.1],    # 右肩胛骨上提范围
'scapula_upward_rot_r' : [-0.190, 0.319], # 右肩胛骨上旋范围

'scapula_abduction_l' :  [-0.628, 0.628], # 左肩胛骨外展范围
'scapula_elevation_l' :  [-0.1, -0.4],    # 左肩胛骨上提范围
'scapula_upward_rot_l' : [-0.210, 0.219], # 左肩胛骨上旋范围 

'elbow_flexion_r' : [0, (3/4)*math.pi],           # 右肘关节屈曲范围
'pro_sup_r'       : [-3/4*math.pi/2, 3/4*math.pi/2], # 右前臂旋前/旋后范围
'wrist_flexion_r' : [-math.pi/2, math.pi/2],      # 右腕关节屈曲范围
'wrist_deviation_r' :[-math.pi/4, math.pi/4],     # 右腕关节偏移范围

'elbow_flexion_l' : [0, (3/4)*math.pi],           # 左肘关节屈曲范围
'pro_sup_l'       : [-math.pi/2, math.pi/2],      # 左前臂旋前/旋后范围
'wrist_flexion_l' : [-math.pi/2, math.pi/2],      # 左腕关节屈曲范围
'wrist_deviation_l' :[-math.pi/4, math.pi/4],     # 左腕关节偏移范围

'shoulder_r_y' : [-math.pi/2, math.pi/2],         # 右肩关节Y轴旋转范围

'lumbar_bending' : [-2/3*math.pi/4, 2/3*math.pi/4], # 腰椎侧屈范围
'lumbar_extension' : [-math.pi/4, math.pi/4],       # 腰椎伸展范围
'lumbar_twist' :  [-math.pi/4, math.pi/4],          # 腰椎扭转范围   

'thorax_bending' :[-math.pi/4, math.pi/4],   # 胸廓侧屈范围
'thorax_extension' :[-math.pi/4, math.pi/4], # 胸廓伸展范围
'thorax_twist' :[-math.pi/4, math.pi/4],     # 胸廓扭转范围

'head_bending' :[-math.pi/4, math.pi/4],     # 头部侧屈范围
'head_extension' :[-math.pi/4, math.pi/4],   # 头部伸展范围
'head_twist' :[-math.pi/4, math.pi/4],       # 头部旋转范围

'ankle_angle_r' : [-math.pi/4, math.pi/4],     # 右踝关节角度范围
'subtalar_angle_r' : [-math.pi/4, math.pi/4],  # 右距下关节角度范围
'mtp_angle_r' : [-math.pi/4, math.pi/4],       # 右跖趾关节角度范围

'ankle_angle_l' : [-math.pi/4, math.pi/4],     # 左踝关节角度范围
'subtalar_angle_l' : [-math.pi/4, math.pi/4],  # 左距下关节角度范围
'mtp_angle_l' : [-math.pi/4, math.pi/4],       # 左跖趾关节角度范围

'knee_angle_r' : [0, 3/4*math.pi],  # 右膝关节角度范围
'knee_angle_l' : [0, 3/4*math.pi],  # 左膝关节角度范围

}

# For each joint of SKEL, we define one corresponding joint of SMPL
# SKEL骨骼与SMPL模型关节的对应关系
# Note that there is not a 1 to 1 correspondance, especially for the arm supination and ankle.
# 注意:这并非完全一一对应,特别是对于手臂旋前/旋后和踝关节
# This correspondance is used to leverage the pose dependant blend shapes from SMPL
# 此对应关系用于利用SMPL模型的姿态依赖形状混合
smpl_joint_corresp = [
0,  # pelvis 骨盆
2,  # femur_r 右股骨 (SMPL中右股骨为关节2)
5,  # tibia_r 右胫骨
8,  # talus_r 右距骨
8,  # calcn_r 右跟骨
11, # toes_r 右脚趾
1,  # femur_l 左股骨
4,  # tibia_l 左胫骨
7,  # talus_l 左距骨
7,  # calcn_l 左跟骨
10, # toes_l 左脚趾
3,  # lumbar_body 腰椎体
6,  # thorax 胸廓
15, # head 头部
14, # scapula_r 右肩胛骨
17, # humerus_r 右肱骨
19, # ulna_r 右尺骨
0,  # radius_r 右桡骨 # 设为0以忽略,因为Rsmpl[0]不用于计算姿态依赖形状混合
21, # hand_r 右手
13, # scapula_l 左肩胛骨
16, # humerus_l 左肱骨
18, # ulna_l 左尺骨
0,  # radius_l 左桡骨
20, # hand_l 左手
]



# Bones scaling 骨骼缩放
""" 
Most bone meshes are scaled given the limb lengths
大多数骨骼网格根据肢体长度进行缩放
For some bones, they need to be scaled wrt the skin. 
对于某些骨骼,需要根据皮肤进行缩放
Given a bone index, for each dimention we give vertex indices to use for scaling
对于给定的骨骼索引,我们为每个维度提供用于缩放的顶点索引
The template bones fit in the template SMPL mesh. So we compare the
模板骨骼适合模板SMPL网格。
vertices distance for the shaped mesh with the vertices distances of the template
因此我们比较有形状的网格的顶点距离与模板的顶点距离
The dimensions represent the different axis:
维度表示不同的轴:
Note that he bones are scaled in the unposed bone space, which is different than SMPL posed space
注意骨骼在无姿态的骨骼空间中缩放,这与SMPL姿态空间不同
0 : front back 前后
1 : up down 上下
2 : left right 左右

(joint index, scaling dimention in bone space, scaling dimention in SMPL space)
(关节索引, 骨骼空间中的缩放维度, SMPL空间中的缩放维度)
"""

scaling_keypoints ={
    # Head
    (13,0,2): (
                410, # between eyes 两眼之间
                384 # back of the head 脑后
            ),
    (13, 1, 1): (
                414, # Top of the head 头顶
                384 # Clavicula hole 锁骨窝
            ),
    (13, 2, 0) : (
                196, # one side head 头部一侧
                3708 # other side 头部另一侧
            ), 
    # Right hand 右手
    (18, 0, 1): (# hand width 手宽
                6179, #r end pinkie 右小指末端
                6137 #r_middle finger end 右中指末端
            ),
    (18, 1, 0): ( # hand length 手长
                5670, #r_wrist_middle,右腕中部
                5906 #r_middle finger end 右中指末端
            ),
    # Left hand 左手
    (23, 0, 1): (# hand width (Use the same vertex references as right hand) 手宽 (使用与右手相同的顶点引用)
                6179, #r end pinkie 右小指末端
                6137 #r_middle finger end 右中指末端
            ),
    (23, 1, 0): ( # hand length 手长
                5670, #r_wrist_middle, 右腕中部
                5906 #r_middle finger end 右中指末端
            ),
    }