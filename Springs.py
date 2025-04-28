from mimetypes import init
from tkinter import N
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

dt = 0.016
n_particles = 100 
n_springs = 200
area = 0.0
L = 10
GRAV = [0.0,-0.0]
WRAP = False
SPRING_K = 30.0
DAMPING = 0.5
RADIUS = 2

particles=np.zeros(n_particles,dtype=[("position", float , 2),
                           ("acceleration", float , 2),
                           ("mass", float),
                           ("velocity", float, 2),
                           ("size", float)])

#particles["position"]=np.random.uniform(0.2,L-0.2,(n_particles,2))

angle_t = np.linspace(0,2*np.pi,n_particles)
for p in range(n_particles):
    particles["position"][p] = np.array([np.cos(angle_t[p])*RADIUS + L/2,np.sin(angle_t[p])*RADIUS + L/2])

particles["mass"] = np.ones(n_particles)
particles["acceleration"] = np.zeros((n_particles,2))
particles["velocity"]=np.zeros((n_particles,2))
particles["size"]=100.0*np.ones(n_particles)

springs =np.zeros(n_springs,dtype=[("particles", int , 2),
                            ("rest", float),
                            ("k", float)])

springs["particles"] = np.zeros((n_springs,2))
springs["rest"] = np.ones(n_springs)*RADIUS*2.0*np.pi/n_particles
springs["k"] = np.ones(n_springs)*SPRING_K

for p in range(len(particles)):
    springs["particles"][p * 2] = [p,(p+1)%len(particles)]
    springs["particles"][p * 2 + 1] = [p,(p+2)%len(particles)]
    springs["rest"][p * 2 + 1] = springs["rest"][p * 2] * 1.6

for p in range(len(particles)):
    to_center = np.array([L/2,L/2]) - particles["position"][p]
    init_dir = np.array([to_center[1],-to_center[0]])
    velocity = init_dir/np.linalg.norm(init_dir)*0.4
    # particles["velocity"][p] = velocity*10.0

particles["velocity"][0] = [-5.0, 5.0]
particles["velocity"][50] = [5.0, -5.0]

fig = plt.figure(figsize=(7,7))
ax = plt.axes(xlim=(0,L),ylim=(0,L))
scatter=ax.scatter(particles["position"][:,0], particles["position"][:,1],s=particles["size"])

def ComputeArea(particles):
    area = 0.0
    for p in range(len(particles)):
        area += np.cross(particles["position"][p],particles["position"][(p+1)%len(particles)])
    return area/2.0

area = ComputeArea(particles)

def ForceField(position, time):
    force_direction = np.array([L/2,L/2])-position
    return force_direction/np.linalg.norm(np.array([L/2,L/2])-position)*1.0

def dot(x0,x1):
    return x0[0]*x1[0]+x0[1]*x1[1]

def ApplySpringForces(particles,springs):
    forces = np.zeros((len(particles),2))
    for s in range(len(springs)):
        p0 = particles["position"][springs["particles"][s][0]]
        p1 = particles["position"][springs["particles"][s][1]]
        v0 = particles["velocity"][springs["particles"][s][0]]
        v1 = particles["velocity"][springs["particles"][s][1]]
        dir = p1 - p0
        dist = np.linalg.norm(dir)
        dir = dir/dist
        force = (springs["k"][s]*(dist/springs["rest"][s] - 1)+DAMPING*(dot(v1-v0,p1-p0)/(springs["rest"][s]*dist)))*dir
        forces[springs["particles"][s][0]] += force
        forces[springs["particles"][s][1]] += -force
    return forces

def compute_normal(particles):
    normals = np.zeros((len(particles),2))
    n = len(particles)
    for i in range(n):
        prev = (i - 1) % n
        next = (i + 1) % n
        d_next = particles["position"][next] - particles["position"][i]
        d_prev = particles["position"][i] - particles["position"][prev]
        tangent = d_next + d_prev
        if np.linalg.norm(tangent) > 0.0:
            tangent = tangent/np.linalg.norm(tangent)
        normals[i] = np.array([-tangent[1],tangent[0]])
    return normals

def update(frame_number):
    forces = ApplySpringForces(particles,springs)

    current_area = ComputeArea(particles)
    area_delta = (current_area - area) / area
    area_force = area_delta * 1000
    normal_dir = compute_normal(particles)
    area_maintain_force = area_force * normal_dir
    forces += area_maintain_force

    for p in range(len(particles)):
        particles["acceleration"][p]= forces[p]/particles["mass"][p] + GRAV
    
    particles["velocity"] = particles["velocity"] + particles["acceleration"]*dt
    particles["position"] = particles["position"] + particles["velocity"]*dt

    if (WRAP):
    #allow wraparound
        particles["position"] = particles["position"]%L
    else:
    #enforce boundary condition
        for p in range(len(particles)):
            if particles["position"][p][0] <= 0.0:
                particles["position"][p][0] = 0.0
                particles["velocity"][p][0] = 0.0
            elif particles["position"][p][0] >= L:
                particles["position"][p][0] = L
                particles["velocity"][p][0] = 0.0

            if particles["position"][p][1] <= 0.0:
                particles["position"][p][1] = 0.0
                particles["velocity"][p][1] = 0.0
            elif particles["position"][p][1] >= L:
                particles["position"][p][1] = L
                particles["velocity"][p][1] = 0.0

    scatter.set_offsets(particles["position"])
    return scatter,

anim = FuncAnimation(fig, update, interval=dt*1000)#interval expects milliseconds but we keep dt in seconds
plt.show()