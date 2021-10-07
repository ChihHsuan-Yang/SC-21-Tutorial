#heat 2d
import deepxde as dde
import numpy as np
# Backend tensorflow.compat.v1 or tensorflow
from deepxde.backend import tf
import time 
import matplotlib.pyplot as plt
import os
from PIL import Image
t0 = time.time()

def plot(geom_time,resolution,data,save_directory,name):  #output_data = pred[:,j]
    img_save_directory = save_directory + 'visualize_result'
    if not os.path.exists(img_save_directory):
        os.makedirs(img_save_directory)
    img_save_directory = img_save_directory + '/'
    fig = plt.figure()
    ims_test = []
    if name[-10:] =='prediction':
        t_max = 1
        t_min = -1
        
    else:
        t_max = np.max(data)
        t_min = np.min(data)
    nx, ny,nt = resolution 
    data = data.reshape((len(data),)) 
    for t in range(nt):
        plt.scatter(geom_time[:,0][nx*ny*t:nx*ny*(t+1)],geom_time[:,1][nx*ny*t:nx*ny*(t+1)], 
            c=data[nx*ny*t:nx*ny*(t+1)].reshape((len(data[nx*ny*t:nx*ny*(t+1)]),)), cmap='jet',vmin=t_min, vmax=t_max, s= 200, marker = 's')
        plt.colorbar()
        plt.xlabel('x domain')
        plt.ylabel('y domain')
        plt.title( 't = ' + "{:.3f}".format(geom_time[:,2][nx*ny*t +1 ]))
        plt.show()
        filename = name + '_' +str(t)
        plt.savefig(os.path.join(img_save_directory, filename + '.png'))
        plt.close()
        im = Image.open(os.path.join(img_save_directory, filename + '.png'))
        ims_test.append(im)    
    ims_test[0].save(os.path.join(img_save_directory + name + '.gif'),save_all = True, 
            append_images = ims_test[1:], optimize = False, duration = 60, loop = 1000)
    im.show()


def plot_mean_data_history(duration, resolution, data,title,save_directory):
    nx,ny,nt = resolution
    m = []
    for t in range(nt):
        mean_t = np.mean(abs(data[nx*ny*t:nx*ny*(t+1)]))
        m.append(mean_t)

    time = np.array(range(nt))*(duration/nt)
    time = time.reshape((nt,1))
    plt.plot(time, np.asarray(m))
    plt.title(title)
    plt.savefig(os.path.join(save_directory, 'mean_' + title + '_history.png'))


def pde(X, u):
    du_X = tf.gradients(u, X)[0]
    du_x, du_y, du_t = du_X[:, 0:1], du_X[:, 1:2],du_X[:, 2:3]
    du_xx  = tf.gradients(du_x, X)[0][:, 0:1]
    du_yy = tf.gradients(du_y, X)[0][:, 1:2]
    return du_t-0.5*(du_xx + du_yy)
    

def func(x):
    return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:2])* np.exp(-x[:, 2:3])


geom = dde.geometry.geometry_2d.Rectangle([-1,-1], [1,1])
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic = dde.IC(geomtime, func, lambda _, on_initial: on_initial)
data = dde.data.TimePDE(
    geomtime,
    pde,
    [],
    num_domain=4000,
    num_boundary=2000,
    num_initial=1000,
    solution=func,
    num_test=1000,
)

layer_size = [3] + [10]+ [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

model.compile("adam", lr=0.001)

t1 = time.time()

losshistory, train_state = model.train(epochs=100)
t2 = time.time()
print("training time")
print(t2-t1)

dde.postprocessing.plot_loss_history(losshistory)
plt.show()


x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
t = np.linspace(0, 1, 21)
test_x , test_t, test_y = np.meshgrid(x, t,y)
test_domain = np.vstack((np.ravel(test_x), np.ravel(test_y),np.ravel(test_t))).T

save_directory = 'C://Users//tcpba/download/'
prediction = model.predict(test_domain)
residual = model.predict(test_domain, operator=pde)
plot_mean_data_history(1, (100,100,21), np.abs(residual),'residual',save_directory)
plot(test_domain,(100,100,21), prediction, save_directory,'prediction')
#plot(test_domain,(100,100,21), residual, save_directory,'prediction')
print("total time")
print(t2-t0)