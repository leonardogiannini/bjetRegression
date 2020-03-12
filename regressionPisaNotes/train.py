import os

if not os.path.exists("./png"):
    os.makedirs("./png")

if not os.path.exists("./h5"):
    os.makedirs("./h5")

if not os.path.exists("./json"):
    os.makedirs("./json")

from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, Flatten, Convolution2D, merge, Convolution1D
from keras.models import Model
from keras.layers import Input
from random import randint
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import roc_curve, auc
import json
import numpy
seed = 7
numpy.random.seed(seed)

inputFile="/scratch/lgiannini/jpsi/featuresData.npz"
f = numpy.load(inputFile)#, allow_pickle=True)

print f.files

inputData = f["arr_0"]

print inputData.shape
inputData=numpy.swapaxes(inputData, 0, 1)

target=inputData[:,0:3]
print target.shape

reg_input=inputData[:,[4,5,6,9,10,11,12,13,14]]
print reg_input.shape


from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(reg_input)

validation_frac=0.2

t_data=reg_input[0:int((1-validation_frac)*len(reg_input))]
v_data=reg_input[int((1-validation_frac)*len(reg_input)):len(reg_input)]

t_target=target[0:int((1-validation_frac)*len(target))]
v_target=target[int((1-validation_frac)*len(target)):len(target)]

scaled_t_data=sc.transform(t_data)
scaled_v_data=sc.transform(v_data)

data_len=scaled_t_data.shape[1]


class wHistory(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}): 
        #if (epoch%0==4):
        self.model.save("h5/normedTraining"+"weights_at_epoch"+str(epoch)+".h5") 
        if epoch==0:
            with open("NetworkJson"+str(epoch)+".json", 'wb') as jsonfile:
                jsonfile.write(self.model.to_json())

        
class plotHistory(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}): 
        #if (epoch%10==9):
            prediction=self.model.predict(scaled_v_data) 
            v_p=(v_target[:,0]**2+v_target[:,1]**2+v_target[:,2]**2)**0.5
            
            plt.clf()
            print prediction.shape
            plt.hist2d(v_p, prediction[:,0],bins=100, norm=LogNorm(), range=[[0,400],[0,400]])
            cb=plt.colorbar()
            plt.xlabel('target of the regression [GeV]')
            plt.ylabel('regressed p [GeV]')
            plt.savefig("png/"+"plot_corrRainbowLog"+str(epoch)+".png")
            cb.remove()
            plt.clf()            
            
            arr1inds = v_p.argsort()
            print arr1inds.shape, "shape"
            sorted_prediction = prediction[arr1inds]
            print "sort"
            plt.hist2d(range(len(sorted_prediction)), sorted_prediction[:,0], bins=100, norm=LogNorm(), range=[[0,len(sorted_prediction)],[0,400]])
            cb=plt.colorbar()
            #plt.plot(sorted_prediction[:,0]) 
            plt.plot(v_p[arr1inds], linewidth=3.0, color="lime")                       
            plt.savefig("png/"+"bias_test"+"_at_epoch"+str(epoch)+".png")
            
            
            prediction=self.model.predict(scaled_t_data) 
            t_p=(t_target[:,0]**2+t_target[:,1]**2+t_target[:,2]**2)**0.5
            
            plt.clf()
            print prediction.shape
            plt.hist2d(t_p, prediction[:,0],bins=100, norm=LogNorm(), range=[[0,400],[0,400]])
            cb=plt.colorbar()
            plt.xlabel('target of the regression [GeV]')
            plt.ylabel('regressed p [GeV]')
            plt.savefig("png/TRAIN"+"plot_corrRainbowLog"+str(epoch)+".png")
            cb.remove()
            plt.clf()            
            
            arr1inds = t_p.argsort()
            print arr1inds.shape, "shape"
            sorted_prediction = prediction[arr1inds]
            print "sort"
            plt.hist2d(range(len(sorted_prediction)), sorted_prediction[:,0], bins=100, norm=LogNorm(), range=[[0,len(sorted_prediction)],[0,400]])
            cb=plt.colorbar()
            #plt.plot(sorted_prediction[:,0]) 
            plt.plot(t_p[arr1inds], linewidth=3.0, color="lime")                       
            plt.savefig("png/TRAIN"+"bias_test"+"_at_epoch"+str(epoch)+".png")



        
wHistory_ = wHistory()
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
pltH_=plotHistory()

dropoutRate=0.2

model = Sequential()

model.add(Dense(50, activation="relu", kernel_initializer="glorot_uniform", input_dim=data_len))
model.add(Dropout(dropoutRate))
model.add(Dense(50, activation="relu", kernel_initializer="glorot_uniform"))
model.add(Dropout(dropoutRate))
model.add(Dense(20, activation="relu", kernel_initializer="glorot_uniform"))
model.add(Dropout(dropoutRate))
model.add(Dense(1, kernel_initializer="glorot_uniform"))
print'compiling'


model.compile(loss='mse', optimizer='adam',metrics=['mse', 'mae'])

model.summary()


print "fitting"

                  
history=model.fit(scaled_t_data , 
                  (t_target[:,0]**2+t_target[:,1]**2+t_target[:,2]**2)**0.5, 
                  nb_epoch=200, verbose=1,batch_size=128, 
                  callbacks=[wHistory_, reduce_lr, pltH_], 
                  validation_data=(scaled_v_data , (v_target[:,0]**2+v_target[:,1]**2+v_target[:,2]**2)**0.5,))

#['loss', 'mean_absolute_error', 'val_mean_squared_error', 'val_mean_absolute_error', 'lr', 'mean_squared_error', 'val_loss']

# list all data in history
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("png/"+"plot_history.png")
plt.clf()


plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("png/"+"plot_history_mse.png")
plt.clf()

plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("png/"+"plot_history_mae.png")
plt.clf()

plt.plot(history.history['lr'])
#plt.plot(history.history['val_mean_absolute_error'])
plt.title('model lr')
plt.ylabel('lr')
plt.xlabel('epoch')
plt.legend(['lr'], loc='upper left')
plt.savefig("png/"+"plot_history_lr.png")
plt.clf()
  
