import taichi as ti
import numpy as np
from math import sqrt
from PIL import Image

import imageio
import trimesh

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# ti.init(arch=ti.cpu)  # or ti.gpu
# ti.init(arch=ti.opengl)  # or ti.gpu
ti.init(arch=ti.cuda)  # or ti.gpu


# subdive 3 ver 642 face 1280 edge 1920 second_edge 930
# subdive 4 ver 2562 face 5120 edge 7680 second_edge 3810
# subdive 5 ver 10242 face 20480 edge 30720 second_edge 15330
# subdive 6 ver 40962 face 81920 edge 122880 second_edge 245730
# subdive 7 ver 163842 face 327680 edge 491520 second_edge 983010


# 模拟参数
subdiv = 4
n_points = 2562 # subdivision=2 的 icosphere 顶点数
n_edges = 7680 # 对应的边数
n_second_edges = 3810 # 对应的边数
n_faces = 5120  # 对应的三角形面数
max_neighbor = 12

dt = 0.01
k_spring = 50.0
damping = 0.05
mass = 0.5
gravity = ti.Vector([0.0, -0.0, 0.0])
rest_len = 0.1  # 默认弹簧长度（可从原始边长算）
IOR = 0.9
DISPERSION = 0.06
THICKNESS_SCAL = 0.1
FRESNEL_RATIO = 0.7
REFLECTANCE_GAMMA_SCALE = 2.0
REFLECTANCE_SCALE = 3.0
GAMMA_CURVE = 50.0
GAMMA_SCALE = 4.5
SIGMOID_CONTRAST = 8.0
wave0 = ti.field(dtype=ti.f32, shape=3)
wave1 = ti.field(dtype=ti.f32, shape=3)

# for camera movement
angle = 0.0
camera_radius = 8.0
origin_point = ti.Vector.field(3, dtype=ti.f32, shape=1)

# 数据结构
x = ti.Vector.field(3, dtype=ti.f32, shape=n_points)  # 位置
v = ti.Vector.field(3, dtype=ti.f32, shape=n_points)  # 速度
f = ti.Vector.field(3, dtype=ti.f32, shape=n_points)  # 力
edges = ti.Vector.field(2, dtype=ti.i32, shape=n_edges)  # 每条边是两个点
second_edges = ti.Vector.field(2, dtype=ti.i32, shape=n_second_edges)  # 每条边是两个点
rest = ti.field(dtype=ti.f32, shape=n_edges)  # 弹簧的初始长度
second_rest = ti.field(dtype=ti.f32, shape=n_second_edges)  # 弹簧的初始长度
face_indexs = ti.Vector.field(3, dtype=ti.i32, shape=n_faces)  # 每个点所属的面
face_indexs_for_render = ti.field(dtype=ti.i32, shape=n_faces * 3)  # 每个点所属的面
colors = ti.Vector.field(3, dtype=ti.f32, shape=n_points)
rds = ti.Vector.field(3, dtype=ti.f32, shape=6)
ti_neighbor_map = ti.field(dtype=ti.i32, shape=(n_points, max_neighbor))

# Texture
image = Image.open("texture.jpg").convert("RGB")
image_np = np.array(image).astype(np.float32) / 255.0
ti_texture = ti.Vector.field(3, dtype=ti.f32, shape=(image_np.shape[1], image_np.shape[0]))
ti_texture.from_numpy(np.transpose(image_np, (1, 0, 2)))

image = Image.open("bw.jpg").convert("RGB")
image_np_bw = np.array(image).astype(np.float32) / 255.0
ti_texture_bw = ti.Vector.field(3, dtype=ti.f32, shape=(image_np.shape[1], image_np.shape[0]))
ti_texture_bw.from_numpy(np.transpose(image_np, (1, 0, 2)))


particles = np.zeros(n_points, dtype=[("position", float , 3)])

origin_edge = ti.Vector.field(2, dtype=ti.f32, shape=n_edges)  # 每条边是两个点
origin_volume = ti.field(dtype=ti.f32, shape=1)
volume = ti.field(dtype=ti.f32, shape=1)
volume_force = ti.field(dtype=ti.f32, shape=1)
mesh_normal = ti.Vector.field(3, dtype=ti.f32, shape=n_points)



def init_speed(edges):
    from collections import defaultdict
    force = 20.0
    tmp_v = np.zeros((n_points, 3) , dtype=float)
    tmp_v[0] = ti.Vector([0.0, force, 0.0])
    tmp_v[500] = ti.Vector([0.0, force, 0.0])
    tmp_v[2000] = ti.Vector([0.0, force, 0.0])

    neighbor_map = defaultdict(set)

    for e in edges:
        a, b = e
        neighbor_map[a].add(b)
        neighbor_map[b].add(a)
    for i in neighbor_map[0]:
        tmp_v[i] = ti.Vector([0.0, 0.8 * force, 0.0])

    for i in neighbor_map[500]:
        tmp_v[i] = ti.Vector([0.0, 0.8 * force, 0.0])
    
    for i in neighbor_map[2000]:
        tmp_v[i] = ti.Vector([0.0, 0.8 * force, 0.0])
    
    v.from_numpy(tmp_v)


# secondary spring

# volume maintain force
@ti.kernel
def compute_volume_maintain_force():
    tmp_v = 0.0
    for i in range(n_faces):
        face_index = face_indexs[i]
        v0 = x[face_index[0]]
        v1 = x[face_index[1]]
        v2 = x[face_index[2]]
        v = ti.math.dot(ti.math.cross(v0, v1), v2) / 6.0
        tmp_v += v
        # print("v", v)
        # print("tmp_v", tmp_v)
    volume[0] = tmp_v
    volume_force[0] = (origin_volume[0] - volume[0]) / origin_volume[0] * 1000
    print("volume_force", volume_force[0])
# normal direction

@ti.kernel
def close_to_neighbor_average():
    print("hello")


@ti.kernel
def compute_forces():
    print("mesh_normal[0]", mesh_normal[0].norm())
    print("volume_force[0]", volume_force[0])
    print("f[0]", (mesh_normal[0] * volume_force[0]).norm())
    for i in range(n_points):
        f[i] = mesh_normal[i] * volume_force[0]

    i, j = edges[0][0], edges[0][1]

    dir = x[j] - x[i]
    dist = dir.norm()

    force = (k_spring * (dist/rest[0] - 1) + damping * ti.math.dot(v[j] - v[i], dir)) * dir.normalized()
 
    for e in range(n_edges):
        i, j = edges[e][0], edges[e][1]
        dir = x[j] - x[i]
        dist = dir.norm()
        force = (k_spring * 2 * (dist/rest[e] - 1) + damping * ti.math.dot(v[j] - v[i], dir)) * dir.normalized()
        f[i] += force
        f[j] -= force

    for e in range(n_second_edges):
        i, j = second_edges[e][0], second_edges[e][1]
        dir = x[j] - x[i]
        dist = dir.norm()
        force = (k_spring * 2 * (dist/second_rest[e] - 1) + damping * ti.math.dot(v[j] - v[i], dir)) * dir.normalized()
        f[i] += force
        f[j] -= force
    

    print("f[0]", f[0].norm())
    # volume maintain force


@ti.kernel
def integrate():
    print(f[0])
    origin_point[0] = ti.Vector([0.0, 0.0, 0.0])
    for i in range(n_points):
        a = f[i] / mass + gravity
        v[i] += dt * a
        x[i] += dt * v[i]
        origin_point[0] += x[i]
    origin_point[0] /= n_points
    
def create_icosphere(subdiv=subdiv):

    mesh = trimesh.creation.icosphere(subdivisions=subdiv, radius=2.0)
    verts = mesh.vertices
    edges_set = set()
    for face in mesh.faces:
        a, b, c = face
        edges_set |= {
            (min(a, b), max(a, b)),
            (min(b, c), max(b, c)),
            (min(c, a), max(c, a)),
        }
    edges_list = list(edges_set)
    print("verts: len", len(verts))
    print("edges_list: len", len(edges_list))
    extra_spring(verts, edges_list)
    return mesh, verts, edges_list

def extra_spring(verts, edges):
    from collections import defaultdict

    neighbor_map = defaultdict(set)
    existing_edges = set()

    for e in edges:
        a, b = e
        neighbor_map[a].add(b)
        neighbor_map[b].add(a)
        existing_edges.add((min(a, b), max(a, b)))
    
    new_edges = set()
    tmp_map = defaultdict(set)
    for i in range(len(verts)):
        n1 = neighbor_map[i]
        for j in n1:
            n2 = neighbor_map[j]
            for k in n2:
                if k != i and k not in n1:
                    edge = (min(i, k), max(i, k))
                    if edge not in existing_edges and edge not in new_edges:
                        tmp_map[edge[0]].add(edge[1])
                        new_edges.add(edge)
    arr = np.array(list(new_edges))
    print("arr: len", len(arr))
    second_edges.from_numpy(arr)


@ti.kernel
def init_rest():
    for i in range(n_edges):
        rest[i] = (x[edges[i][0]] - x[edges[i][1]]).norm()
    for i in range(n_second_edges):
        second_rest[i] = (x[second_edges[i][0]] - x[second_edges[i][1]]).norm()

@ti.func
def contrast(c):
    return 1.0 / (1.0 + ti.math.exp(-SIGMOID_CONTRAST * (c - 0.5)))
    

@ti.func
def fancy_cube(n):
    colx = ti.Vector([0.0, 0.0, 0.0])
    coly = ti.Vector([0.0, 0.0, 0.0])
    colz = ti.Vector([0.0, 0.0, 0.0])
    if n[0] != 0.0:
        colx = ti_texture[int(0.5 + THICKNESS_SCAL*n[1]/n[0]), int(0.5 + THICKNESS_SCAL*n[2]/n[0])]
    if n[1] != 0.0:
        coly = ti_texture[int(0.5 + THICKNESS_SCAL*n[2]/n[1]), int(0.5 + THICKNESS_SCAL*n[0]/n[1])]
    if n[2] != 0.0:
        colz = ti_texture[int(0.5 + THICKNESS_SCAL*n[0]/n[2]), int(0.5 + THICKNESS_SCAL*n[1]/n[2])]

    t = n * n
    sam = (colx * t[0] + coly * t[1] + colz * t[2])/(t[0] + t[1] + t[2])

    return sam


@ti.func
def filmic_gamma(i):
    return ti.math.log(GAMMA_CURVE * i + 1.0) / GAMMA_SCALE


@ti.func
def filmic_gamma_inverse(i):
    return (1.0 / GAMMA_CURVE) * (ti.math.exp(GAMMA_SCALE * i) - 1.0)


@ti.func
def texture_3d_load(view_dir):
    abs_d = ti.abs(view_dir)
    uv = ti.Vector([0.0, 0.0])
    if abs_d[0] > abs_d[1] and abs_d[0] > abs_d[2]:
        if view_dir[0] > 0:
            uv = 0.5 * ti.Vector([-view_dir[2], -view_dir[1]])/ abs_d[0] + 0.5
        else:
            uv = 0.5 * ti.Vector([view_dir[2], -view_dir[1]])/ abs_d[0] + 0.5
    elif abs_d[1] > abs_d[2]:
        if view_dir[1] > 0:
            uv = 0.5 * ti.Vector([-view_dir[0], view_dir[2]])/ abs_d[1] + 0.5
        else:
            uv = 0.5 * ti.Vector([view_dir[0], -view_dir[2]])/ abs_d[1] + 0.5
    else:
        if view_dir[2] > 0:
            uv = 0.5 * ti.Vector([view_dir[0], -view_dir[1]])/ abs_d[2] + 0.5
        else:
            uv = 0.5 * ti.Vector([-view_dir[0], -view_dir[1]])/ abs_d[2] + 0.5
    color = ti_texture_bw[int(uv[0] * (ti_texture_bw.shape[0]-1)), int(uv[1] * (ti_texture_bw.shape[1]-1))]
    return color

@ti.func
def sampleWeights(i):
    return ti.Vector([(1.0 - i) * (1.0 - i), 2.8 * i * (1.0 - i), i * i])

@ti.func
def texCubeSampleWeights(i):
    w = ti.Vector([(1.0 - i) * (1.0 - i), 2.8 * i * (1.0 - i), i * i])
    return w / ti.math.dot(w, ti.Vector([1.0, 1.0, 1.0]))

@ti.func
def simpleampleCubeMap(w, rrd):
    t = texture_3d_load(rrd)
    col = ti.Vector([t[0], t[1], t[2]])
    re = ti.Vector([ti.math.dot(texCubeSampleWeights(w[0]), col), ti.math.dot(texCubeSampleWeights(w[1]), col), ti.math.dot(texCubeSampleWeights(w[2]), col)])
    return re

@ti.func
def sampleCubeMap(wave, rds0, rds1, rds2):
    col0 = texture_3d_load(rds0)
    col1 = texture_3d_load(rds1)
    col2 = texture_3d_load(rds2)
    return ti.Vector([ti.math.dot(texCubeSampleWeights(wave[0]), col0), ti.math.dot(texCubeSampleWeights(wave[1]), col1), ti.math.dot(texCubeSampleWeights(wave[2]), col2)])

@ti.func
def resample(wave0, wave1, intensity0, intensity1):
    w0 = sampleWeights(wave0[0])
    w1 = sampleWeights(wave0[1])
    w2 = sampleWeights(wave0[2])
    w3 = sampleWeights(wave1[0])
    w4 = sampleWeights(wave1[1])
    w5 = sampleWeights(wave1[2])
    return intensity0[0] * w0  + intensity0[1] * w1 + intensity0[2] * w2 + intensity1[0] * w3 + intensity1[1] * w4 + intensity1[2] * w5
    

@ti.func
def resample_color(rds, refl0, refl1, wave0, wave1):
    cube0 = sampleCubeMap(wave0, rds[0], rds[1], rds[2])
    cube1 = sampleCubeMap(wave1, rds[3], rds[4], rds[5])
    intensity0 = filmic_gamma_inverse(cube0) + refl0
    intensity1 = filmic_gamma_inverse(cube1) + refl1
    col = resample(wave0, wave1, intensity0, intensity1)
    return 1.4 * filmic_gamma(col / 6.0)

# @ti.func
# def resample_color(rds, refl0, refl1, wave0, wave1):
#     intensity0 = refl0
#     intensity1 = refl1
#     col = resample(wave0, wave1, intensity0, intensity1)
#     return 1.4 * filmic_gamma(col / 6.0)



@ti.kernel
def update_color(camera_pos: ti.math.vec3):
    iors0 = IOR + wave0 * DISPERSION
    iors1 = IOR + wave1 * DISPERSION

    for i in range(n_points):
    # for i in range(1):
        # print("break 0")
        n = mesh_normal[i].normalized()
        view_dir = (camera_pos - x[i]).normalized()

        # print("break 1")
        # print("n", n)
        # print("break 2")

        sam = fancy_cube(n)
        filmThickness = sam[0] + 0.1

        att0 = 0.5 + 0.5 * ti.math.cos(((THICKNESS_SCAL * filmThickness) / (wave0 + 1.0)) * ti.math.dot(n, view_dir))
        att1 = 0.5 + 0.5 * ti.math.cos(((THICKNESS_SCAL * filmThickness) / (wave1 + 1.0)) * ti.math.dot(n, view_dir))
        # print("break 3")
        rior0 = 1.0 / iors0
        rior1 = 1.0 / iors1
        t0 = ti.math.pow((1.0 - rior0)/(1.0 + rior0), ti.Vector([2.0, 2.0, 2.0]))
        t1 = ti.math.pow((1.0 - rior1)/(1.0 + rior1), ti.Vector([2.0, 2.0, 2.0]))
        tt = ti.math.pow(ti.math.clamp(1.0 + ti.math.dot(n, view_dir), 0.0, 1.0), 5.0)
        f0 = (1.0 - FRESNEL_RATIO) + FRESNEL_RATIO * (t0 + (1.0 - t0) * tt)
        f1 = (1.0 - FRESNEL_RATIO) + FRESNEL_RATIO * (t1 + (1.0 - t1) * tt)

        rrd = view_dir - 2.0 * ti.math.dot(n, view_dir) * n

        # print("break 4")
        cube0 = REFLECTANCE_GAMMA_SCALE * att0 * simpleampleCubeMap(wave0, rrd)
        cube1 = REFLECTANCE_GAMMA_SCALE * att1 * simpleampleCubeMap(wave1, rrd)

        refl0 = REFLECTANCE_SCALE * filmic_gamma_inverse(ti.math.mix(ti.Vector([0.0, 0.0, 0.0]), cube0, f0))
        refl1 = REFLECTANCE_SCALE * filmic_gamma_inverse(ti.math.mix(ti.Vector([0.0, 0.0, 0.0]), cube1, f1))
        # print("break 5")

        rds[0] = ti.math.refract(view_dir, n, iors0[0])
        rds[1] = ti.math.refract(view_dir, n, iors0[1])
        rds[2] = ti.math.refract(view_dir, n, iors0[2])
        rds[3] = ti.math.refract(view_dir, n, iors1[0])
        rds[4] = ti.math.refract(view_dir, n, iors1[1])
        rds[5] = ti.math.refract(view_dir, n, iors1[2])

        col = resample_color(rds, refl0, refl1, wave0, wave1)
        col = contrast(col)

        # cos_theta = ti.math.dot(n, view_dir)
        # fresnel = ti.pow(1.0 - cos_theta, 5.0)
        # wave = ti.sin(50 * cos_theta) * 0.5 + 0.5

        # r = 0.5 * fresnel + 0.5 * wave
        # g = 0.3 * fresnel + 0.7 * wave
        # b = 1.0 * (1.0 - fresnel) * wave

        # r = 1.0 / (1.0 - ti.math.exp( -0.8 * (r - 0.5)))
        # g = 1.0 / (1.0 - ti.math.exp( -0.8 * (g - 0.5)))
        # b = 1.0 / (1.0 - ti.math.exp( -0.8 * (b - 0.5)))

        colors[i] = col
    # print("inside update_color")
    # print("color", colors[0])

# 初始化
mesh, verts_np, edges_np = create_icosphere()
edges_np = np.array(edges_np[:n_edges])  # 截取前 n_edges 条
print("edges_np: len", len(edges_np))
print("verts_np: len", len(verts_np))
print("face: len", len(mesh.faces))

x.from_numpy(verts_np[:n_points])
v.fill([0.0, 0.0, 0.0])
edges.from_numpy(edges_np[:n_edges])
face_indexs.from_numpy(mesh.faces[:n_faces])
face_indexs_for_render.from_numpy(mesh.faces[:n_faces].reshape(-1))
compute_volume_maintain_force()
origin_volume[0] = volume[0]
init_rest()
particles["position"] = x.to_numpy()
wave0 = ti.Vector([1.0, 0.8, 0.6])
wave1 = ti.Vector([0.4, 0.2, 0.0])

def update(frame_number):
    # mesh.vertex_normals = None
    mesh_normal.from_numpy(np.array(mesh.vertex_normals[:n_points]))
    # print("mesh_normal", mesh_normal[0])
    compute_volume_maintain_force()
    compute_forces()
    integrate()
    mesh.vertices[:] = x.to_numpy()
    particles["position"] = x.to_numpy()
    scatter._offsets3d = (particles["position"][:, 0], particles["position"][:, 1], particles["position"][:, 2])
    return scatter,

@ti.kernel
def test():
    dir = x[1] - x[0]
    dist = dir.norm()
    dir = dir / dist

@ti.kernel
def bary():

    print("hello")

def compute_barycentric(v0, v1, v2, p):
    # v0, v1, v2: 顶点坐标
    # p: hit点坐标
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = p - v0

    d00 = np.dot(v0v1, v0v1)
    d01 = np.dot(v0v1, v0v2)
    d11 = np.dot(v0v2, v0v2)
    d20 = np.dot(v0p, v0v1)
    d21 = np.dot(v0p, v0v2)

    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return u, v, w

test()

init_speed(edges_np)


# bubble = ti.ui.Mesh(vertices=x, indices=face_indexs)

mesh_normal.from_numpy(np.array(mesh.vertex_normals[:n_points]))
# update_color(ti.Vector([5, 5, 5]))
# print("color outside", colors[0])

frames = []
frame_count = 0
total_frames = 60 * 30


compute_volume_maintain_force()
compute_forces()
integrate()
mesh.vertices[:] = x.to_numpy()
mesh_normal.from_numpy(np.array(mesh.vertex_normals[:n_points]))

cam_x = camera_radius * ti.math.cos(angle) 
cam_y = 0.0
cam_z = camera_radius * ti.math.sin(angle)

lookat_point = origin_point[0].to_numpy()
camera_pos = np.array([cam_x, cam_y, cam_z]) + lookat_point
width = 1024
height = 1024

import ray

ray_origin, ray_direction = ray.generate_camera_rays(camera_pos, lookat_point, width, height)
print(ray_origin.shape)
print(ray_direction.shape)
print(type(mesh.ray))
print("hello")

import pyembree
print(pyembree)

locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origin, ray_direction, multiple_hits=False)

print(locations.shape)
print(index_ray.shape)
print(index_tri.shape)


def compute_barycentric_batch(v0s, v1s, v2s, ps):
    v0v1 = v1s - v0s
    v0v2 = v2s - v0s
    v0p = ps - v0s

    d00 = np.sum(v0v1 * v0v1, axis=1)
    d01 = np.sum(v0v1 * v0v2, axis=1)
    d11 = np.sum(v0v2 * v0v2, axis=1)
    d20 = np.sum(v0p * v0v1, axis=1)
    d21 = np.sum(v0p * v0v2, axis=1)

    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return u, v, w

def fancy_cube_batch(n):
    N = n.shape[0]

    colx = ti.Vector([0.0, 0.0, 0.0])
    coly = ti.Vector([0.0, 0.0, 0.0])
    colz = ti.Vector([0.0, 0.0, 0.0])

    n0 = n[:, 0]
    n1 = n[:, 1]
    n2 = n[:, 2]

    t = n * n  # (N, 3)

    eps = 1e-6

    # Prepare uv for each channel
    # colx: (n[1]/n[0], n[2]/n[0])
    mask_x = np.abs(n0) > eps
    uvx_0 = 0.5 + THICKNESS_SCAL * (n1 / (n0 + (~mask_x) * eps))
    uvx_1 = 0.5 + THICKNESS_SCAL * (n2 / (n0 + (~mask_x) * eps))

    # coly: (n[2]/n[1], n[0]/n[1])
    mask_y = np.abs(n1) > eps
    uvy_0 = 0.5 + THICKNESS_SCAL * (n2 / (n1 + (~mask_y) * eps))
    uvy_1 = 0.5 + THICKNESS_SCAL * (n0 / (n1 + (~mask_y) * eps))

    # colz: (n[0]/n[2], n[1]/n[2])
    mask_z = np.abs(n2) > eps
    uvz_0 = 0.5 + THICKNESS_SCAL * (n0 / (n2 + (~mask_z) * eps))
    uvz_1 = 0.5 + THICKNESS_SCAL * (n1 / (n2 + (~mask_z) * eps))

    H, W, _ = image_np.shape

    # Clip UV到合法区间，防止越界
    uvx_0 = np.clip((uvx_0 * (W-1)).astype(np.int32), 0, W-1)
    uvx_1 = np.clip((uvx_1 * (H-1)).astype(np.int32), 0, H-1)

    uvy_0 = np.clip((uvy_0 * (W-1)).astype(np.int32), 0, W-1)
    uvy_1 = np.clip((uvy_1 * (H-1)).astype(np.int32), 0, H-1)

    uvz_0 = np.clip((uvz_0 * (W-1)).astype(np.int32), 0, W-1)
    uvz_1 = np.clip((uvz_1 * (H-1)).astype(np.int32), 0, H-1)

    # Sample color
    colx = np.zeros((N, 3), dtype=np.float32)
    coly = np.zeros((N, 3), dtype=np.float32)
    colz = np.zeros((N, 3), dtype=np.float32)

    colx[mask_x] = image_np[uvx_1[mask_x], uvx_0[mask_x]]
    coly[mask_y] = image_np[uvy_1[mask_y], uvy_0[mask_y]]
    colz[mask_z] = image_np[uvz_1[mask_z], uvz_0[mask_z]]

    # Weighted sum
    numerator = colx * t[:, 0:1] + coly * t[:, 1:2] + colz * t[:, 2:3]
    denominator = (t[:, 0:1] + t[:, 1:2] + t[:, 2:3]) + eps
    print("numerator", numerator.shape)
    print("denominator", denominator.shape)

    output = numerator / denominator
    print("output", output.shape)

    return output  # (N, 3)

def fresnel(rd, norm, n2):
    r0 = np.power((1.0 - n2) / (1.0 + n2), np.array(2.0, 2.0, 2.0))
    return r0 + (1.0 - r0) * np.power(np.clip(1.0 - np.sum(rd * norm, axis=1), 0.0, 1.0), 5.0)

def texture_3d_load_batch(view_dirs):
    N = view_dirs.shape[0]
    abs_d = np.abs(view_dirs)
    
    # 初始化UV坐标数组
    uvs = np.zeros((N, 2), dtype=np.float32)
    
    # 创建三种情况的掩码
    mask_x_major = (abs_d[:, 0] > abs_d[:, 1]) & (abs_d[:, 0] > abs_d[:, 2])
    mask_y_major = (abs_d[:, 1] > abs_d[:, 2]) & ~mask_x_major
    mask_z_major = ~mask_x_major & ~mask_y_major
    
    # X轴主导情况
    x_pos = view_dirs[:, 0] > 0
    x_pos_mask = mask_x_major & x_pos
    x_neg_mask = mask_x_major & ~x_pos
    
    # X正向
    if np.any(x_pos_mask):
        uvs[x_pos_mask, 0] = 0.5 - 0.5 * view_dirs[x_pos_mask, 2] / abs_d[x_pos_mask, 0]  # -z/x
        uvs[x_pos_mask, 1] = 0.5 - 0.5 * view_dirs[x_pos_mask, 1] / abs_d[x_pos_mask, 0]  # -y/x
    
    # X负向
    if np.any(x_neg_mask):
        uvs[x_neg_mask, 0] = 0.5 + 0.5 * view_dirs[x_neg_mask, 2] / abs_d[x_neg_mask, 0]  # z/x
        uvs[x_neg_mask, 1] = 0.5 - 0.5 * view_dirs[x_neg_mask, 1] / abs_d[x_neg_mask, 0]  # -y/x
    
    # Y轴主导情况
    y_pos = view_dirs[:, 1] > 0
    y_pos_mask = mask_y_major & y_pos
    y_neg_mask = mask_y_major & ~y_pos
    
    # Y正向
    if np.any(y_pos_mask):
        uvs[y_pos_mask, 0] = 0.5 - 0.5 * view_dirs[y_pos_mask, 0] / abs_d[y_pos_mask, 1]  # -x/y
        uvs[y_pos_mask, 1] = 0.5 + 0.5 * view_dirs[y_pos_mask, 2] / abs_d[y_pos_mask, 1]  # z/y
    
    # Y负向
    if np.any(y_neg_mask):
        uvs[y_neg_mask, 0] = 0.5 + 0.5 * view_dirs[y_neg_mask, 0] / abs_d[y_neg_mask, 1]  # x/y
        uvs[y_neg_mask, 1] = 0.5 - 0.5 * view_dirs[y_neg_mask, 2] / abs_d[y_neg_mask, 1]  # -z/y
    
    # Z轴主导情况
    z_pos = view_dirs[:, 2] > 0
    z_pos_mask = mask_z_major & z_pos
    z_neg_mask = mask_z_major & ~z_pos
    
    # Z正向
    if np.any(z_pos_mask):
        uvs[z_pos_mask, 0] = 0.5 + 0.5 * view_dirs[z_pos_mask, 0] / abs_d[z_pos_mask, 2]  # x/z
        uvs[z_pos_mask, 1] = 0.5 - 0.5 * view_dirs[z_pos_mask, 1] / abs_d[z_pos_mask, 2]  # -y/z
    
    # Z负向
    if np.any(z_neg_mask):
        uvs[z_neg_mask, 0] = 0.5 - 0.5 * view_dirs[z_neg_mask, 0] / abs_d[z_neg_mask, 2]  # -x/z
        uvs[z_neg_mask, 1] = 0.5 - 0.5 * view_dirs[z_neg_mask, 1] / abs_d[z_neg_mask, 2]  # -y/z
    
    # 转换UV坐标到图像索引
    H, W, _ = image_np_bw.shape
    u_indices = np.clip((uvs[:, 0] * (W - 1)).astype(np.int32), 0, W - 1)
    v_indices = np.clip((uvs[:, 1] * (H - 1)).astype(np.int32), 0, H - 1)
    
    # 采样颜色
    colors = image_np_bw[v_indices, u_indices]
    
    return colors

def sampleWeights_batch(i):
    return np.array([(1.0 - i) * (1.0 - i), 2.8 * i * (1.0 - i), i * i])

def texCubeSampleWeights_batch(i):
    w = np.array([(1.0 - i) * (1.0 - i), 2.8 * i * (1.0 - i), i * i])
    return w / np.sum(w)

def simpleampleCubeMap_batch(w, rrd):
    # 使用批处理版本的texture_3d_load函数
    t = texture_3d_load_batch(rrd)  # 形状(N, 3)
    
    # 对每个采样结果应用权重
    weights0 = texCubeSampleWeights_batch(w[0])  # 形状(3,)
    weights1 = texCubeSampleWeights_batch(w[1])  # 形状(3,)
    weights2 = texCubeSampleWeights_batch(w[2])  # 形状(3,)
    
    # 计算加权结果，使用einsum进行批量点积运算
    result0 = np.einsum('i,ni->n', weights0, t)  # 形状(N,)
    result1 = np.einsum('i,ni->n', weights1, t)  # 形状(N,)
    result2 = np.einsum('i,ni->n', weights2, t)  # 形状(N,)
    
    # 合并结果
    return np.column_stack([result0, result1, result2])  # 形状(N, 3)

def sampleCubeMap_batch(wave, rds0, rds1, rds2):
    # 分别对三个方向采样
    col0 = texture_3d_load_batch(rds0)  # 形状(N, 3)
    col1 = texture_3d_load_batch(rds1)  # 形状(N, 3)
    col2 = texture_3d_load_batch(rds2)  # 形状(N, 3)
    
    # 计算权重
    weights0 = texCubeSampleWeights_batch(wave[0])  # 形状(3,)
    weights1 = texCubeSampleWeights_batch(wave[1])  # 形状(3,)
    weights2 = texCubeSampleWeights_batch(wave[2])  # 形状(3,)
    
    # 计算加权结果
    result0 = np.einsum('i,ni->n', weights0, col0)  # 形状(N,)
    result1 = np.einsum('i,ni->n', weights1, col1)  # 形状(N,)
    result2 = np.einsum('i,ni->n', weights2, col2)  # 形状(N,)
    
    # 合并结果
    return np.column_stack([result0, result1, result2])  # 形状(N, 3)


def render(hit_locations, hit_index_ray, hit_index_tri):
    wave0 = np.array([1.0, 0.8, 0.6])
    wave1 = np.array([0.4, 0.2, 0.0])

    iors0 = IOR + wave0 * DISPERSION
    iors1 = IOR + wave1 * DISPERSION


    face_indices = mesh.faces[hit_index_tri]  # (N, 3)，每行是一个三角形的3个顶点索引

    v0s = mesh.vertices[face_indices[:, 0]]  # (N, 3)，取每个face的第一个点
    v1s = mesh.vertices[face_indices[:, 1]]  # (N, 3)
    v2s = mesh.vertices[face_indices[:, 2]]  # (N, 3)

    n0s = mesh.vertex_normals[face_indices[:, 0]]  # 对应的normal
    n1s = mesh.vertex_normals[face_indices[:, 1]]
    n2s = mesh.vertex_normals[face_indices[:, 2]]

    u, v, w = compute_barycentric_batch(v0s, v1s, v2s, hit_locations)

    # here got all barycentric normals
    bary_normals = u[:, None] * n0s + v[:, None] * n1s + w[:, None] * n2s

    sam = fancy_cube_batch(bary_normals)
    filmThickness = sam[:, 0] + 0.1
    filmThickness = filmThickness[:, np.newaxis]
    print("filmThickness", filmThickness.shape)

    dot_products = np.sum(bary_normals * ray_direction[hit_index_ray], axis=1)
    dot_products = dot_products[:, np.newaxis]

    att0 = 0.5 + 0.5 * np.cos(((THICKNESS_SCAL * filmThickness) / (wave0 + 1.0)) * dot_products)
    att1 = 0.5 + 0.5 * np.cos(((THICKNESS_SCAL * filmThickness) / (wave1 + 1.0)) * dot_products)

    rior0 = 1.0 / iors0
    rior1 = 1.0 / iors1

    f0 = (1.0 - FRESNEL_RATIO) + FRESNEL_RATIO * fresnel(ray_direction[hit_index_ray], bary_normals, iors0)
    f1 = (1.0 - FRESNEL_RATIO) + FRESNEL_RATIO * fresnel(ray_direction[hit_index_ray], bary_normals, iors1)

    rrd = ray_direction[hit_index_ray] - 2.0 * dot_products * bary_normals

    cube0 = REFLECTANCE_GAMMA_SCALE * att0 * texture_3d_load_batch(rrd)

    # for i in range(len(hit_locations)):
    #     n = mesh_normal[i].normalized()
    #     view_dir = (camera_pos - x[i]).normalized()

    #     sam = fancy_cube(n)
    #     filmThickness = sam[0] + 0.1

    #     att0 = 0.5 + 0.5 * ti.math.cos(((THICKNESS_SCAL * filmThickness) / (wave0 + 1.0)) * ti.math.dot(n, view_dir))
    #     att1 = 0.5 + 0.5 * ti.math.cos(((THICKNESS_SCAL * filmThickness) / (wave1 + 1.0)) * ti.math.dot(n, view_dir))
    #     # print("break 3")
    #     rior0 = 1.0 / iors0
    #     rior1 = 1.0 / iors1
    #     t0 = ti.math.pow((1.0 - rior0)/(1.0 + rior0), ti.Vector([2.0, 2.0, 2.0]))
    #     t1 = ti.math.pow((1.0 - rior1)/(1.0 + rior1), ti.Vector([2.0, 2.0, 2.0]))
    #     tt = ti.math.pow(ti.math.clamp(1.0 + ti.math.dot(n, view_dir), 0.0, 1.0), 5.0)
    #     f0 = (1.0 - FRESNEL_RATIO) + FRESNEL_RATIO * (t0 + (1.0 - t0) * tt)
    #     f1 = (1.0 - FRESNEL_RATIO) + FRESNEL_RATIO * (t1 + (1.0 - t1) * tt)

    #     rrd = view_dir - 2.0 * ti.math.dot(n, view_dir) * n

    #     # print("break 4")
    #     cube0 = REFLECTANCE_GAMMA_SCALE * att0 * simpleampleCubeMap(wave0, rrd)
    #     cube1 = REFLECTANCE_GAMMA_SCALE * att1 * simpleampleCubeMap(wave1, rrd)

    #     refl0 = REFLECTANCE_SCALE * filmic_gamma_inverse(ti.math.mix(ti.Vector([0.0, 0.0, 0.0]), cube0, f0))
    #     refl1 = REFLECTANCE_SCALE * filmic_gamma_inverse(ti.math.mix(ti.Vector([0.0, 0.0, 0.0]), cube1, f1))
    #     # print("break 5")

    #     rds[0] = ti.math.refract(view_dir, n, iors0[0])
    #     rds[1] = ti.math.refract(view_dir, n, iors0[1])
    #     rds[2] = ti.math.refract(view_dir, n, iors0[2])
    #     rds[3] = ti.math.refract(view_dir, n, iors1[0])
    #     rds[4] = ti.math.refract(view_dir, n, iors1[1])
    #     rds[5] = ti.math.refract(view_dir, n, iors1[2])

    #     col = resample_color(rds, refl0, refl1, wave0, wave1)
    #     col = contrast(col)

        # cos_theta = ti.math.dot(n, view_dir)
        # fresnel = ti.pow(1.0 - cos_theta, 5.0)
        # wave = ti.sin(50 * cos_theta) * 0.5 + 0.5

        # r = 0.5 * fresnel + 0.5 * wave
        # g = 0.3 * fresnel + 0.7 * wave
        # b = 1.0 * (1.0 - fresnel) * wave

        # r = 1.0 / (1.0 - ti.math.exp( -0.8 * (r - 0.5)))
        # g = 1.0 / (1.0 - ti.math.exp( -0.8 * (g - 0.5)))
        # b = 1.0 / (1.0 - ti.math.exp( -0.8 * (b - 0.5)))

        # colors[i] = col



render(locations, index_ray, index_tri)






