#import numpy as np
#import matplotlib.pyplot as plt


import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Slider, Button



V = np.array([[0 ,0.2,0.28752689], [0, 0.2,0.2236], [0, 0.287228,0.2],[0, 0.1,0.23]])

points = np.array([[0,0,0]])
for i,vector in enumerate(V): points = np.append(points,[points[i]+vector],axis=0)

def angle_dot(a, b):
    dot_product = np.dot(a, b)
    prod_of_norms = np.linalg.norm(a) * np.linalg.norm(b)
    return np.arccos(dot_product / prod_of_norms)




alpha = angle_dot([0,1,0],V[0])
beta = angle_dot(V[0],V[1])
gamma = angle_dot(V[1],V[2])
delta = angle_dot(V[2],V[3])
print(alpha,beta,gamma,delta)

fig, ax = plt.subplots(1, 1)
ax = plt.axes(projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1) 
ax.set_zlim(-1, 1)

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.subplots_adjust(left=0.25, bottom=0.4)
plt.grid(visible=True)
l,=plt.plot(points[:,0], points[:,1],'.',linestyle='-', linewidth=2, markersize=12)


axbase = plt.axes([0.25, 0.3, 0.65, 0.03])
axa = plt.axes([0.25, 0.25, 0.65, 0.03])
axb  = plt.axes([0.25, 0.2, 0.65, 0.03])
axc = plt.axes([0.25, 0.15, 0.65, 0.03])
axd  = plt.axes([0.25, 0.1, 0.65, 0.03])


base= Slider(axbase, 'base', 0.0 , 2*np.pi, valinit=0)
alpha= Slider(axa, 'alpha', 0.0 , 2*np.pi, valinit=alpha)
beta = Slider(axb, 'beta', 0.00, 2*np.pi, valinit=beta)
gamma = Slider(axc, 'gamma', 0.00, 2*np.pi, valinit=gamma)
delta = Slider(axd, 'delta', 0.00, 2*np.pi, valinit=delta)
def update(val):
    z_rot = np.array([[1 ,0 ,0,0],[0, np.cos(base.val),-np.sin(base.val),0],[0, np.sin(base.val),np.cos(base.val),0],[0, 0,0,1]]) #matriz de rotação para o plano xOy (visto de cima)
    
    points = np.array([[0,0,0]])
    beta_ = beta.val-angle_dot([0,1,0],V[0])
    gamma_ = gamma.val-angle_dot([0,1,0],V[1])
    delta_ = delta.val-angle_dot([0,1,0],V[2])
    #print(alpha,beta,gamma,delta)
    #print(np.linalg.norm(V[3]))
    V[0] = [0,np.linalg.norm(V[0])*np.cos(alpha.val),np.linalg.norm(V[0])*np.sin(alpha.val)]
    V[1] = [0,np.linalg.norm(V[1])*np.cos(beta_),np.linalg.norm(V[1])*np.sin(beta_)]
    V[2] = [0,np.linalg.norm(V[2])*np.cos(gamma_),np.linalg.norm(V[2])*np.sin(gamma_)]
    V[3] = [0,np.linalg.norm(V[3])*np.cos(delta_),np.linalg.norm(V[3])*np.sin(delta_)]
    print(np.linalg.norm(V[0]),np.linalg.norm(V[0])*np.cos(alpha.val)**2+np.linalg.norm(V[0])*np.sin(alpha.val)**2,V[0])
    for i,vector in enumerate(V): points = np.append(points,[points[i]+vector],axis=0)
    print(points)
    #print(points,np.amax(points[:,1]))
    #print(points.T.shape)
    
    #rotação
    points = np.insert(points.T, 0, [1,1,1,1,1], axis=0)
    points = np.matmul(z_rot, points)
    points = np.delete(points, 0, 0)
    points = points.T
    print(points)
    
    l.set_xdata(points[1:,0])
    l.set_ydata(points[1:,1])
    l.set_3d_properties(points[1:,2])
    fig.canvas.draw_idle()
base.on_changed(update)
alpha.on_changed(update)
beta.on_changed(update)
gamma.on_changed(update)
delta.on_changed(update)
plt.show()


