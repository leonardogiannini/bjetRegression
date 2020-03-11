import os

if not os.path.exists("./png"):
    os.makedirs("./png")

if not os.path.exists("./h5"):
    os.makedirs("./h5")

if not os.path.exists("./json"):
    os.makedirs("./json")

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, merge, LSTM, Input
from keras.layers.normalization import BatchNormalization

import matplotlib, json, numpy, glob
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from random import randint
seed = 7
numpy.random.seed(seed)#init for reproducibilty 

data_len=4
all_len=5

def Generator(files):
    print "generator"
    #shuffle(files)
    while 1:
        for f in files:
            print f
            print files.index(f)
            
            jet=numpy.load(f)['arr_0'][:,[1,2,0,3,4]]
            PFcs=numpy.load(f)['arr_1'][:,:,:]
            SVs=numpy.load(f)['arr_2'][:,:,[0,1,3,4,5,6,7,8]]
            
            print jet.shape,PFcs.shape,SVs.shape

            mean=numpy.array([8.21603851e+01,  -5.64793451e-03, 2.14345341e+01, 8.21733246e+01,   8.77616043e+01])
            std=numpy.array([50.84579849,   1.16241825, 7.74034548,  50.62704086,  53.44575119])

            jet2=(jet[:]-mean)/std

            print jet.shape
            print PFcs.shape

            t_data=[jet,PFcs[:, ::-1], SVs[:, ::-1]]
            scaled_t_data=[jet2[:,0:3],PFcs[:, ::-1] ,SVs[:, ::-1]]
            
            print "Yield"
            yield scaled_t_data,t_data[0][:,data_len:all_len]/t_data[0][:,0:1]
            print "just yielded"
            

def GeneratorV(files):
    print "generator"
    #shuffle(files)
    while 1:
        for f in files:
            print f
            print files.index(f)
            
            jet=numpy.load(f)['arr_0'][:,[1,2,0,3,4]]
            PFcs=numpy.load(f)['arr_1'][:,:,:]
            SVs=numpy.load(f)['arr_2'][:,:,[0,1,3,4,5,6,7,8]]
            
            print jet.shape,PFcs.shape,SVs.shape

            mean=numpy.array([8.21603851e+01,  -5.64793451e-03, 2.14345341e+01, 8.21733246e+01,   8.77616043e+01])
            std=numpy.array([50.84579849,   1.16241825, 7.74034548,  50.62704086,  53.44575119])
            
            jet2=(jet[:]-mean)/std

            print jet.shape
            print PFcs.shape

            t_data=[jet,PFcs[:, ::-1], SVs[:, ::-1]]
            scaled_t_data=[jet2[:,0:3],PFcs[:, ::-1] ,SVs[:, ::-1]]
            
            print "Yield validation"
            yield scaled_t_data,t_data[0][:,data_len:all_len]/t_data[0][:,0:1]
            print "just yielded validation"


class wHistory(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}): 
        self.model.save("h5/normedTraining"+"weights_at_epoch"+str(epoch)+".h5") 
        if epoch==0:
            with open("NetworkJson"+str(epoch)+".json", 'wb') as jsonfile:
                jsonfile.write(self.model.to_json())
        os.system("python Draw_Plots_LSTM.py "+"h5/normedTraining"+"weights_at_epoch"+str(epoch)+".h5"+" "+str(epoch)+"  &")
     
wHistory_ = wHistory()
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

dropoutRate=0.1

Inputs=[Input(shape=(3,)) ]
Inputs+=[Input(shape=(100,7))]
Inputs+=[Input(shape=(5,8))]


pfLSTM=LSTM(50)
svLSTM=LSTM(20)

pfvector=Inputs[1]

#pfvector = BatchNormalization()(pfvector)
pfLSTM = pfLSTM(pfvector)
pfLSTM = Dense(50, activation='relu',kernel_initializer='lecun_uniform')(pfLSTM)
pfLSTM = Dense(50, activation='relu',kernel_initializer='lecun_uniform')(pfLSTM)


svvector=Inputs[2]

#svvector = BatchNormalization()(svvector)
svLSTM = svLSTM(svvector)
svLSTM = Dense(20, activation='relu',kernel_initializer='lecun_uniform')(svLSTM)
svLSTM = Dense(20, activation='relu',kernel_initializer='lecun_uniform')(svLSTM)

x = merge( [Inputs[0] , pfLSTM, svLSTM] , mode='concat')

x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
x = Dropout(dropoutRate)(x)
x = Dense(50, activation='relu',kernel_initializer='lecun_uniform')(x)
x = Dropout(dropoutRate)(x)

predictions = Dense(1,kernel_initializer='lecun_uniform')(x)
model = Model(inputs=Inputs, outputs=predictions)
print'compiling'

model.compile(loss='mse', optimizer='adam',metrics=['mse', 'mae'])
model.summary()

##Train this model

allfiles=glob.glob("/gpfs/ddn/cms/user/lgiannini/DeepNtupleRegression/batches/*npz")
files=allfiles[0:1300]
v_files=allfiles[1300:1428]

print len(files), len(v_files)

history = model.fit_generator(Generator(files), steps_per_epoch=1300, epochs=100, verbose=1,callbacks=[wHistory_,reduce_lr],validation_data=GeneratorV(v_files),validation_steps=128)

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
 
            
            
