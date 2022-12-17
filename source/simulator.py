import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Slider, Button


draw_points = np.array([0, 0 ,0, 1])

fig = plt.figure()
ax = fig.add_subplot(1, 1,1, projection='3d')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2) 
ax.set_zlim(-0, 2)

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.subplots_adjust(left=0.25, bottom=0.4)
plt.grid(visible=True)
l,=plt.plot([0], [0],'.',linestyle='-', linewidth=2, markersize=12)
l2,=plt.plot([0], [0], [0], markerfacecolor='k', markeredgecolor='k' ,marker='o', markersize=2, alpha=0.6,linestyle = 'None')

axbase = plt.axes([0.25, 0.3, 0.65, 0.03])
axa = plt.axes([0.25, 0.25, 0.65, 0.03])
axb  = plt.axes([0.25, 0.2, 0.65, 0.03])
axc = plt.axes([0.25, 0.15, 0.65, 0.03])


base= Slider(axbase, 'base', 0.0 , 2*np.pi, valinit=0)
alpha_= Slider(axa, 'alpha', -np.pi/2 , np.pi/2, valinit=0)
beta = Slider(axb, 'beta',-np.pi/2 , np.pi/2, valinit=0)
gamma = Slider(axc, 'gamma', 0.00, 2*np.pi, valinit=0)
def update(val):
    global draw_points 
    #matrizes de transformação
    alpha,a,d,teta = (0,0,0.3585,base.val)
    t10 = np.array([[np.cos(teta), -np.sin(teta), 0,a],[np.cos(alpha)*np.sin(teta), np.cos(alpha)*np.cos(teta), -np.sin(alpha),-np.sin(alpha)*d],[np.sin(alpha)*np.sin(teta), np.sin(alpha)*np.cos(teta),np.cos(alpha),np.cos(alpha)*d],[0, 0, 0,1]])
    alpha,a,d,teta = (np.pi/2+alpha_.val,0,0.300,0)
    t21= np.array([[np.cos(teta), -np.sin(teta), 0,a],[np.cos(alpha)*np.sin(teta), np.cos(alpha)*np.cos(teta), -np.sin(alpha),-np.sin(alpha)*d],[np.sin(alpha)*np.sin(teta), np.sin(alpha)*np.cos(teta),np.cos(alpha),np.cos(alpha)*d],[0, 0, 0,1]])
    alpha,a,d,teta = (beta.val,0,0.350,0)
    t32 = np.array([[np.cos(teta), -np.sin(teta), 0,a],[np.cos(alpha)*np.sin(teta), np.cos(alpha)*np.cos(teta), -np.sin(alpha),-np.sin(alpha)*d],[np.sin(alpha)*np.sin(teta), np.sin(alpha)*np.cos(teta),np.cos(alpha),np.cos(alpha)*d],[0, 0, 0,1]])
    alpha,a,d,teta = (gamma.val,0,0.251,0)
    t43 = np.array([[np.cos(teta), -np.sin(teta), 0,a],[np.cos(alpha)*np.sin(teta), np.cos(alpha)*np.cos(teta), -np.sin(alpha),-np.sin(alpha)*d],[np.sin(alpha)*np.sin(teta), np.sin(alpha)*np.cos(teta),np.cos(alpha),np.cos(alpha)*d],[0, 0, 0,1]])
    
    points = np.array([[0,0,0,1]])
    t20 = np.matmul(t10,t21)
    t30 = np.matmul(t20,t32)
    t40 = np.matmul(t30,t43)

    points = np.vstack((points,t10.dot(np.array([0,0,0,1]))))
    points = np.vstack((points,t21.dot(np.array([0,0,0,1]))))
    points = np.vstack((points,t32.dot(np.array([0,0,0,1]))))
    points = np.vstack((points,t43.dot(np.array([0,0,0,1]))))

    
    points[2] = np.matmul((t20),np.array([0,0,0,1]))
    points[3] = np.matmul((t30),np.array([0,0,0,1]))
    points[4] = np.matmul((t40),np.array([0,0,0,1]))
    print(points)
    if ((points[4][2] < 0.01) and (points[4][2] > -0.01)):
        draw_points = np.vstack((draw_points,points[4]))
        l2.set_xdata(draw_points[:,0])
        l2.set_ydata(draw_points[:,1])
        l2.set_3d_properties([0])
    l.set_xdata(points[:,0])
    l.set_ydata(points[:,1])
    l.set_3d_properties(points[:,2])
    fig.canvas.draw_idle()

base.on_changed(update)
alpha_.on_changed(update)
beta.on_changed(update)
gamma.on_changed(update)
plt.show()
