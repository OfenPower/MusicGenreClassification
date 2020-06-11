import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras

# generate data 
num_data = 200 #number of samples per class

linSep=False

if linSep:

    # linear separable
    mean0 = np.transpose([-1, -1])
    X0 = mean0 + 0.1*np.random.randn(num_data,2)
    Y0 = np.ones((num_data,2))
    Y0[:,1] = 0*Y0[:,1]

    mean1 = np.transpose([0.5, 0.5])
    X1 = mean1 + 0.1*np.random.randn(num_data,2)
    Y1 = np.ones((num_data,2))
    Y1[:,0] = 0*Y1[:,0]

    X = np.concatenate((X0, X1))
    Y = np.concatenate((Y0, Y1))

    X=X.astype(np.float32)
    Y=Y.astype(np.float32)
    X=np.reshape(X,(2*num_data,2,))

else:

    # circular nested
    mean0 = np.transpose([0, 0])
    X0 = mean0 + 0.1*np.random.randn(num_data,2)
    Y0 = np.ones((num_data,2))
    Y0[:,1] = 0*Y0[:,1]
    
    
    # points on a circle 
    rand_ang=2*np.pi*np.random.random((num_data,1))
    radius=1
    X_ang=radius*(np.concatenate((np.cos(rand_ang),np.sin(rand_ang)),axis=1))
    print(np.shape(X_ang))
    
    mean1 = np.transpose([0.0, 0.0])
    X1 = mean1 + 0.1*np.random.randn(num_data,2)
    X1=X1+X_ang
    Y1 = np.ones((num_data,2))
    Y1[:,0] = 0*Y1[:,0]

X = np.concatenate((X0, X1))
Y = np.concatenate((Y0, Y1))

X=X.astype(np.float32)
Y=Y.astype(np.float32)
X=np.reshape(X,(2*num_data,2,))

fig, ax = plt.subplots(figsize=(12, 12))  
         
ax.scatter(X[0:num_data,0], X[0:num_data,1], c='blue',  label='train class 0',
               alpha=1)
ax.scatter(X[num_data+1:2*num_data,0], X[num_data+1:2*num_data,1], c='green',  
               label='train class 1',alpha=1)

ax.legend(fontsize=20)
ax.grid(True)
plt.show()


#plot decision region

def plot_decision_regions(model,X,Y):
    
   
    num_rand_points=5000
    rand_points=3*(np.random.random((num_rand_points,2) )-[0.5, 0.5])
    pred=(model(rand_points))
    ind_0=np.where(np.argmax(pred,axis=1)==0)
    ind_1=np.where(np.argmax(pred,axis=1)==1)

    
    fig, ax = plt.subplots(figsize=(15, 12))  
    
    ax.scatter(rand_points[ind_0,0], rand_points[ind_0,1], c='tab:blue',  label='class 0',
                   alpha=0.2, edgecolors='tab:blue')
    ax.scatter(rand_points[ind_1,0], rand_points[ind_1,1], c='tab:green',  label='class 1',
                   alpha=0.2, edgecolors='tab:green')
        
    ax.scatter(X[0:num_data,0], X[0:num_data,1], c='blue',  label='train class 0',
               alpha=1)
    ax.scatter(X[num_data+1:2*num_data,0], X[num_data+1:2*num_data,1], c='green',  
               label='train class 1',alpha=1)

    ax.legend(fontsize=20)
    ax.grid(True)
    plt.show()


# -------------------------------------------------------------------------------------------------------------------------------------

# application different SGD variants
print(tf.__version__)


# -----------------setup model

model = keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu',input_shape=(2,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
model.summary()



opt_sgd = tf.keras.optimizers.SGD(lr=0.01)
opt_sgd_mom = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.5, nesterov=False)
opt_adam = tf.keras.optimizers.Adam(lr=0.01)

model.compile(optimizer=opt_adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])
#----train model
model.fit(X,Y,
          batch_size=20,
          epochs=12,
          validation_split=0.2)


#-----show decision region
plot_decision_regions(model,X,Y)
