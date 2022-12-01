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
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2) 
ax.set_zlim(-0, 2)

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
alpha_= Slider(axa, 'alpha', -np.pi/2 , np.pi/2, valinit=alpha)
beta = Slider(axb, 'beta',-np.pi/2 , np.pi/2, valinit=beta)
gamma = Slider(axc, 'gamma', 0.00, 2*np.pi, valinit=gamma)
delta = Slider(axd, 'delta', 0.00, 2*np.pi, valinit=delta)
def update(val):
    alpha,a,d,teta = (0,0,0.3585,base.val)
    t01 = np.array([[np.cos(teta), -np.sin(teta), 0,a],[np.cos(alpha)*np.sin(teta), np.cos(alpha)*np.cos(teta), -np.sin(alpha),-np.sin(alpha)*d],[np.sin(alpha)*np.sin(teta), np.sin(alpha)*np.cos(teta),np.cos(alpha),np.cos(alpha)*d],[0, 0, 0,1]])
    alpha,a,d,teta = (np.pi/2+alpha_.val,0,0.300,0)
    t12= np.array([[np.cos(teta), -np.sin(teta), 0,a],[np.cos(alpha)*np.sin(teta), np.cos(alpha)*np.cos(teta), -np.sin(alpha),-np.sin(alpha)*d],[np.sin(alpha)*np.sin(teta), np.sin(alpha)*np.cos(teta),np.cos(alpha),np.cos(alpha)*d],[0, 0, 0,1]])
    alpha,a,d,teta = (beta.val,0,0.350,0)
    t23 = np.array([[np.cos(teta), -np.sin(teta), 0,a],[np.cos(alpha)*np.sin(teta), np.cos(alpha)*np.cos(teta), -np.sin(alpha),-np.sin(alpha)*d],[np.sin(alpha)*np.sin(teta), np.sin(alpha)*np.cos(teta),np.cos(alpha),np.cos(alpha)*d],[0, 0, 0,1]])
    alpha,a,d,teta = (gamma.val,0,0.251,0)
    t34 = np.array([[np.cos(teta), -np.sin(teta), 0,a],[np.cos(alpha)*np.sin(teta), np.cos(alpha)*np.cos(teta), -np.sin(alpha),-np.sin(alpha)*d],[np.sin(alpha)*np.sin(teta), np.sin(alpha)*np.cos(teta),np.cos(alpha),np.cos(alpha)*d],[0, 0, 0,1]])
    transformations = np.array([t01,t12,t23,t34])
    points = np.array([[0,0,0,1]])
    
    for i,transf in enumerate(transformations): 
        print(points)
        points = np.append(points,[np.dot(transf,points[i])],axis=0)

    alpha,a,d,teta = (0,0,0.0,base.val)
    t01 = np.array([[np.cos(teta), -np.sin(teta), 0,a],[np.cos(alpha)*np.sin(teta), np.cos(alpha)*np.cos(teta), -np.sin(alpha),-np.sin(alpha)*d],[np.sin(alpha)*np.sin(teta), np.sin(alpha)*np.cos(teta),np.cos(alpha),np.cos(alpha)*d],[0, 0, 0,1]])
    
    points = t01.dot(points.T)
    points = points.T
    print(points)
    l.set_xdata(points[:,0])
    l.set_ydata(points[:,1])
    l.set_3d_properties(points[:,2])
    fig.canvas.draw_idle()

base.on_changed(update)
alpha_.on_changed(update)
beta.on_changed(update)
gamma.on_changed(update)
delta.on_changed(update)
plt.show()
