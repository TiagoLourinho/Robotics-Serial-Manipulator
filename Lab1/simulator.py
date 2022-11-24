#import numpy as np
#import matplotlib.pyplot as plt


import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Slider, Button



V = np.array([[0.5,2], [1,0.5], [1,-1],[1,1]])
points = np.array([[0,0]])
for i,vector in enumerate(V): points = np.append(points,[points[i]+vector],axis=0)

def angle_dot(a, b):
    dot_product = np.dot(a, b)
    prod_of_norms = np.linalg.norm(a) * np.linalg.norm(b)
    return np.arccos(dot_product / prod_of_norms)




alpha = angle_dot([1,0],V[0])
beta = angle_dot(V[0],V[1])
gamma = angle_dot(V[1],V[2])
delta = angle_dot(V[2],V[3])
print(alpha,beta,gamma,delta)

fig, ax = plt.subplots(1, 1)
plt.ylim([0, 10])
plt.subplots_adjust(left=0.25, bottom=0.4)
plt.grid(visible=True)
l,=plt.plot(points[:,0], points[:,1],'.',linestyle='-', linewidth=2, markersize=12)

axa = plt.axes([0.25, 0.25, 0.65, 0.03])
axb  = plt.axes([0.25, 0.2, 0.65, 0.03])
axc = plt.axes([0.25, 0.15, 0.65, 0.03])
axd  = plt.axes([0.25, 0.1, 0.65, 0.03])

alpha= Slider(axa, 'alpha', 0.0 , 2*np.pi, valinit=alpha)
beta = Slider(axb, 'beta', 0.00, 2*np.pi, valinit=beta)
gamma = Slider(axc, 'gamma', 0.00, 2*np.pi, valinit=gamma)
delta = Slider(axd, 'delta', 0.00, 2*np.pi, valinit=delta)
def update(val):
    
    points = np.array([[0,0]])
    beta_ = (beta.val-angle_dot([1,0],V[0]))
    gamma_ = gamma.val-angle_dot([1,0],V[1])
    delta_ = delta.val-angle_dot([1,0],V[2])
    #print(alpha,beta,gamma,delta)
    print(np.linalg.norm(V[3]))
    V[0] = [np.linalg.norm(V[0])*np.cos(alpha.val),np.linalg.norm(V[0])*np.sin(alpha.val)]
    V[1] = [np.linalg.norm(V[1])*np.cos(beta_),np.linalg.norm(V[1])*np.sin(beta_)]
    V[2] = [np.linalg.norm(V[2])*np.cos(gamma_),np.linalg.norm(V[2])*np.sin(gamma_)]
    V[3] = [np.linalg.norm(V[3])*np.cos(delta_),np.linalg.norm(V[3])*np.sin(delta_)]
    print(np.linalg.norm(V[3]),delta_)
    for i,vector in enumerate(V): points = np.append(points,[points[i]+vector],axis=0)
    #print(points,np.amax(points[:,1]))
    
    l.set_ydata(points[:,1])
    l.set_xdata(points[:,0])
    fig.canvas.draw_idle()
alpha.on_changed(update)
beta.on_changed(update)
gamma.on_changed(update)
delta.on_changed(update)
plt.show()


