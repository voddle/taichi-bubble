import numpy as np
import taichi as ti

def generate_camera_rays(camera_pos, camera_lookat, width, height, fov_y=45.0):
    camera_pos = np.array(camera_pos)
    camera_lookat = np.array(camera_lookat)

    # Step 1: 相机坐标系
    forward = camera_lookat - camera_pos
    forward = forward / np.linalg.norm(forward)
    
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    
    up = np.cross(right, forward)

    # Step 2: 屏幕参数
    aspect_ratio = width / height
    fov_y_rad = np.deg2rad(fov_y)
    scale = np.tan(fov_y_rad * 0.5)
    
    # Step 3: 创建网格（像素坐标）
    i = np.arange(width)
    j = np.arange(height)
    u, v = np.meshgrid(i, j)
    
    # 归一化到[-1, 1]
    u = (u + 0.5) / width * 2 - 1
    v = (v + 0.5) / height * 2 - 1

    # Step 4: 计算每个像素的方向
    dirs = forward.reshape(1, 1, 3) \
         + u[:, :, np.newaxis] * right.reshape(1, 1, 3) * aspect_ratio * scale \
         + v[:, :, np.newaxis] * up.reshape(1, 1, 3) * scale
    
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)  # 单位化
    
    # Step 5: flatten成一维
    ray_origins = np.broadcast_to(camera_pos, (height, width, 3)).reshape(-1, 3)
    ray_directions = dirs.reshape(-1, 3)

    return ray_origins, ray_directions



