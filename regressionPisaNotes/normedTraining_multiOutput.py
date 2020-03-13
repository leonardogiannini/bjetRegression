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
numpy.random.seed(seed)#init for reproducibilty


#data handling

def Draw_vars(regression_vars, prefix="prefix", titles=["v1"]):
    for i in range(len(regression_vars[0])):
        print prefix+"regression_var"+str(i)
        plt.clf()
        plt.hist(regression_vars[:,i], histtype='bar', stacked=True, bins=100, log=False, color="c",  edgecolor="none", alpha=1)
        plt.title(prefix+titles[i])
        plt.grid(True, which='both')
        plt.savefig("png/"+prefix+titles[i]+".png")
        plt.clf()
        plt.hist(regression_vars[:,i], histtype='bar', stacked=True, bins=100, log=True, color="lime",  edgecolor="none", alpha=1)
        plt.title(prefix+titles[i])
        plt.grid(True, which='both')
        plt.savefig("png/"+prefix+titles[i]+"log.png")
        plt.clf()

regTree=numpy.load("/gpfs/ddn/cms/user/lgiannini/RegressionTrees/Tree_training.npz")['arr_0']

vars={'nPVs':0,'Jet_vtxNtrk':1,'Jet_vtxMass':2,'Jet_vtx3dL':3,'Jet_vtx3deL':4,
'Jet_vtxPt':5,'dR':6,'Jet_puId':7,'Jet_btagCSV':8,'Jet_rawPt':9,
'Jet_corr':10,'Jet_mcPt':11,'Jet_mcPtq':12,'Jet_mcFlavour':13,'Jet_pt':14,'Jet_ptd':15,'Jet_mt':16,
'Jet_eta':17,'Jet_phi':18,'Jet_mass':19,'Jet_chHEF':20,'Jet_neHEF':21,'Jet_chEmEF':22,
'Jet_neEmEF':23,'Jet_chMult':24,'Jet_leadTrackPt':25,'Jet_mcEta':26,'Jet_mcPhi':27,
'Jet_mcM':28,'Jet_mcE':29,'Jet_leptonPt':30,'Jet_leptonPtRel':31,'Jet_leptonPtRelInv':32,
'Jet_leptonDeltaR':33,'Jet_leptonDeltaEta':34,'Jet_leptonDeltaPhi':35,'rho':36,
'met_pt':37,'met_phi':38,'Jet_met_dPhi':39,'Jet_met_proj':40, 'Jet_leptonPdgId':41, 'Jet_pt_reg':42 }

# factory->AddVariable( "Jet_pt", "Jet_pt", "units", 'F' );
#//factory->AddVariable( "Jet_rawPt", "Jet_rawPt", "units", 'F' );
#factory->AddVariable( "nPVs", "nPVs", "units", 'F' );
#factory->AddVariable( "Jet_eta", "Jet_eta", "units", 'F' );
#factory->AddVariable( "Jet_mt", "Jet_mt", "units", 'F' );
#factory->AddVariable("Jet_area","Jet_area","units",'F');
#//factory->AddVariable( "Jet_leadTrackPt  ", "Jet_leadTrackPt  ", "units", 'F' );
#factory->AddVariable( "Jet_leptonPtRel","Jet_leptonPtRel","units",'F');
#factory->AddVariable( "Jet_leptonPt","Jet_leptonPt","units",'F');
#factory->AddVariable( "Jet_leptonDeltaR","Jet_leptonDeltaR","units",'F');
#//factory->AddVariable( "Jet_neHEF", "Jet_neHEF" , "units", 'F' );
#//factory->AddVariable( "Jet_neEmEF", "Jet_neEmEF", "units", 'F' );
#factory->AddVariable( "Jet_vtxPt", "Jet_vtxPt", "units", 'F' );
#factory->AddVariable( "Jet_vtxMass", "Jet_vtxMass", "units", 'F' );
#factory->AddVariable( "Jet_vtx3dL", "Jet_vtx3dL", "units", 'F' );
#factory->AddVariable( "Jet_vtxNtrk", "Jet_vtxNtrk", "units", 'I' );
#factory->AddVariable( "Jet_vtx3deL", "Jet_vtx3deL", "units", 'F' );
   
new_listofVars=[
    'Jet_pt','Jet_mt','Jet_eta','Jet_mass','Jet_corr','Jet_ptd',
    'Jet_chHEF','Jet_neHEF','Jet_chEmEF','Jet_neEmEF','Jet_leadTrackPt','nPVs',
    'Jet_vtxNtrk','Jet_vtxMass','Jet_vtx3dL','Jet_vtx3deL','Jet_vtxPt',
    'Jet_leptonPt','Jet_leptonPtRel','Jet_leptonPtRelInv','Jet_leptonDeltaR','Jet_leptonPdgId',
    "Jet_mcPt", "Jet_pt_reg"
    ]

data_len=len(new_listofVars)-2
all_len=len(new_listofVars)-1

#regression_inputs=[vars["Jet_pt"],vars["nPVs"],vars["Jet_eta"],vars["Jet_mt"],
       #vars["Jet_area"],vars["Jet_leptonPtRel"],vars["Jet_leptonPt"],vars["Jet_leptonDeltaR"],
       #vars["Jet_vtxPt"],vars["Jet_vtxMass"],vars["Jet_vtx3dL"],vars["Jet_vtxNtrk"],vars["Jet_vtx3deL"],
       #vars["Jet_mcPt"]]

regression_inputs=[vars[var] for var in new_listofVars]

print regression_inputs

regTree=regTree[:,regression_inputs]
print regTree.shape
print regTree[0]

#regTree=regTree[regTree[:,0]<200]
#print regTree.shape
regTree=regTree[regTree[:,0]>20]
print regTree.shape
regTree=regTree[abs(regTree[:,2])<2.5]
print regTree.shape
regTree=regTree[regTree[:,data_len]>20]
print regTree.shape
regTree=regTree[regTree[:,all_len]>0]
print regTree.shape

numpy.random.shuffle(regTree)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(regTree[:,0:data_len])

validation_frac=0.1

t_data=regTree[0:int((1-validation_frac)*len(regTree))]
v_data=regTree[int((1-validation_frac)*len(regTree)):len(regTree)]

print v_data[v_data[:,0]<50].shape
print v_data.shape

scaled_t_data=sc.transform(t_data[:,0:data_len])
scaled_v_data=sc.transform(v_data[:,0:data_len])


#Draw_vars(t_data, "data_",new_listofVars )
#Draw_vars(scaled_t_data, "normed_data_", new_listofVars)
#Draw_vars(v_data, "v_data_", new_listofVars)
#Draw_vars(scaled_v_data, "normed_v_data_", new_listofVars)



#keras part

from keras import backend as K

def mean_squared_error_rel(y_true, y_pred):
    return K.mean(K.square((y_pred - y_true)/y_true), axis=-1)

def tilted_loss_2(y,f):
    q = 0.75
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

def tilted_loss_1(y,f):
    q = 0.25
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

class wHistory(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}): 
        #if (epoch%0==4):
        self.model.save("h5/normedTraining"+"weights_at_epoch"+str(epoch)+".h5") 
        if epoch==0:
            with open("NetworkJson"+str(epoch)+".json", 'wb') as jsonfile:
                jsonfile.write(self.model.to_json())
        #string_name="eval3Test_at_epoch"+str(epoch)+".h5"
        
class plotHistory(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}): 
        #if (epoch%10==9):
            prediction_all=self.model.predict(scaled_v_data) 
            print len(prediction_all), " sahpe" 
            prediction=prediction_all[0]*v_data[:,0:1]
            predictionUp=prediction_all[1]*v_data[:,0:1]
            predictionDown=prediction_all[2]*v_data[:,0:1]
            
            plt.clf()
            print prediction.shape
            plt.hist2d(v_data[:,data_len], prediction[:,0],bins=100, norm=LogNorm(), range=[[0,400],[0,400]])
            cb=plt.colorbar()
            plt.xlabel('target of the regression [GeV]')
            plt.ylabel('regressed pT [GeV]')
            plt.savefig("png/"+"plot_corrRainbowLog"+str(epoch)+".png")
            cb.remove()
            plt.clf()
            
            
            print prediction[:,0].shape, prediction[:,0:1].shape, v_data[:,data_len:all_len].shape
            plt.hist(v_data[:,data_len:all_len]/prediction[:,0:1], bins=200, range=[-0.5,2.5], alpha=1, color='red', edgecolor='red', label="mc/regressed", histtype='stepfilled')
            plt.hist(v_data[:,data_len:all_len]/v_data[:,data_len+1:all_len+1], bins=200, range=[-0.5,2.5], alpha=0.5, color='gold', edgecolor='gold', label="mc/old_reg", histtype='stepfilled')
            plt.hist(v_data[:,data_len:all_len]/v_data[:,0:1], bins=200, range=[-0.5,2.5], alpha=0.5, color='blue', edgecolor='blue', label="mc/pt", histtype='step')
            plt.legend(loc=1)
            plt.grid(True, which='both')
            plt.xlabel('pT MC / pT')
            plt.ylabel('# evts.')
            plt.savefig("png/"+"plot_ratio"+"_at_epoch"+str(epoch)+".png")
            plt.clf()
            
            
            print prediction[:,0].shape, prediction[:,0:1].shape, v_data[:,data_len:all_len].shape
            plt.hist((prediction[:,0:1]-v_data[:,data_len:all_len])/v_data[:,data_len:all_len], bins=200, range=[-1,1], alpha=1, color='red', edgecolor='red', label="regressed-mc / mc", histtype='stepfilled')
            plt.hist((v_data[:,data_len+1:all_len+1]-v_data[:,data_len:all_len])/v_data[:,data_len:all_len], bins=200, range=[-1,1], alpha=0.5, color='gold', edgecolor='gold', label="old_reg-mc / mc", histtype='stepfilled')
            plt.hist((v_data[:,0:1]-v_data[:,data_len:all_len])/v_data[:,data_len:all_len], bins=200, range=[-1,1], alpha=0.5, color='blue', edgecolor='blue', label="pt-mc / mc", histtype='step')
            plt.legend(loc=1)
            plt.grid(True, which='both')
            plt.xlabel('Rel pT difference')
            plt.ylabel('# evts.')
            plt.savefig("png/"+"plot_RelativeDifference"+"_at_epoch"+str(epoch)+".png")
            plt.clf()
            
            #difference in ranges 0-50
            inds=(v_data[:,0]<50)*(v_data[:,all_len]>0)
            inds2=(v_data[:,0]>50)*(v_data[:,0]<100)*(v_data[:,all_len]>0)
            inds3=(v_data[:,0]>100)*(v_data[:,all_len]>0)
            
            plt.hist(v_data[inds][:,data_len:all_len]/prediction[inds][:,0:1], bins=200, range=[-0.5,2.5], alpha=1, color='red', edgecolor='red', label="mc/regressed", histtype='stepfilled')
            plt.hist(v_data[inds][:,data_len:all_len]/predictionDown[inds][:,0:1], bins=200, range=[-0.5,2.5], alpha=1, color='red', edgecolor='red', label="up", histtype='step')
            plt.hist(v_data[inds][:,data_len:all_len]/predictionUp[inds][:,0:1], bins=200, range=[-0.5,2.5], alpha=1, color='red', edgecolor='red', label="down", histtype='step')
            plt.hist(v_data[inds][:,data_len:all_len]/v_data[inds][:,data_len+1:all_len+1], bins=200, range=[-0.5,2.5], alpha=0.5, color='gold', edgecolor='gold', label="mc/old_reg", histtype='stepfilled')
            plt.hist(v_data[inds][:,data_len:all_len]/v_data[inds][:,0:1], bins=200, range=[-0.5,2.5], alpha=0.5, color='blue', edgecolor='blue', label="mc/pt", histtype='step')
            plt.legend(loc=1)
            plt.grid(True, which='both')
            plt.xlabel('pT MC / pT range [0,50] GeV')
            plt.ylabel('# evts.')
            plt.savefig("png/"+"plot_difference_0-50"+"_at_epoch"+str(epoch)+".png")
            plt.clf()
            
            plt.hist(v_data[inds2][:,data_len:all_len]/prediction[inds2][:,0:1], bins=200, range=[-0.5,2.5], alpha=1, color='red', edgecolor='red', label="mc/regressed", histtype='stepfilled')
            plt.hist(v_data[inds2][:,data_len:all_len]/predictionDown[inds2][:,0:1], bins=200, range=[-0.5,2.5], alpha=1, color='red', edgecolor='red', label="up", histtype='step')
            plt.hist(v_data[inds2][:,data_len:all_len]/predictionUp[inds2][:,0:1], bins=200, range=[-0.5,2.5], alpha=1, color='red', edgecolor='red', label="down", histtype='step')
            plt.hist(v_data[inds2][:,data_len:all_len]/v_data[inds2][:,data_len+1:all_len+1], bins=200, range=[-0.5,2.5], alpha=0.5, color='gold', edgecolor='gold', label="mc/old_reg", histtype='stepfilled')
            plt.hist(v_data[inds2][:,data_len:all_len]/v_data[inds2][:,0:1], bins=200, range=[-0.5,2.5], alpha=0.5, color='blue', edgecolor='blue', label="mc/pt", histtype='step')
            plt.legend(loc=1)
            plt.grid(True, which='both')
            plt.xlabel('pT MC / pT range [50,100] GeV')
            plt.ylabel('# evts.')
            plt.savefig("png/"+"plot_difference_50-100"+"_at_epoch"+str(epoch)+".png")
            plt.clf()
            
            plt.hist(v_data[inds3][:,data_len:all_len]/prediction[inds3][:,0:1], bins=200, range=[-0.5,2.5], alpha=1, color='red', edgecolor='red', label="mc/regressed", histtype='stepfilled')
            plt.hist(v_data[inds3][:,data_len:all_len]/predictionDown[inds3][:,0:1], bins=200, range=[-0.5,2.5], alpha=1, color='red', edgecolor='red', label="up", histtype='step')
            plt.hist(v_data[inds3][:,data_len:all_len]/predictionUp[inds3][:,0:1], bins=200, range=[-0.5,2.5], alpha=1, color='red', edgecolor='red', label="down", histtype='step')
            plt.hist(v_data[inds3][:,data_len:all_len]/v_data[inds3][:,data_len+1:all_len+1], bins=200, range=[-0.5,2.5], alpha=0.5, color='gold', edgecolor='gold', label="mc/old_reg", histtype='stepfilled')
            plt.hist(v_data[inds3][:,data_len:all_len]/v_data[inds3][:,0:1], bins=200, range=[-0.5,2.5], alpha=0.5, color='blue', edgecolor='blue', label="mc/pt", histtype='step')
            plt.legend(loc=1)
            plt.grid(True, which='both')
            plt.xlabel('pT MC / pT range [100,inf] GeV')
            plt.ylabel('# evts.')
            plt.savefig("png/"+"plot_difference_100-Inf"+"_at_epoch"+str(epoch)+".png")
            plt.clf()


            plt.hist(prediction[:,0:1], bins=100, range=[0,400], alpha=1, color='red', edgecolor='red', label="regressed new", histtype='step', log=True)
            plt.hist(v_data[:,data_len+1:all_len+1], bins=100, range=[0,400], alpha=1, color='gold', edgecolor='gold', label="pt old reg", histtype='step', log=True)
            plt.hist(v_data[:,data_len:all_len], bins=100, range=[0,400], alpha=1, color='blue', edgecolor='blue', label="pt mc", histtype='step', log=True)
            plt.hist(v_data[:,0:1], bins=100, range=[0,400], alpha=1, color='green', edgecolor='green', label="pt", histtype='step', log=True)
            plt.legend(loc=1)
            plt.grid(True, which='both')
            plt.xlabel('pT [GeV]')
            plt.ylabel('# evts.')
            plt.savefig("png/plot_pTspectrum"+"_at_epoch"+str(epoch)+"_log.png")
            plt.clf()
            
            plt.hist(prediction[:,0:1], bins=100, range=[0,400], alpha=1, color='red', edgecolor='red', label="regressed new", histtype='step', log=False)
            plt.hist(v_data[:,data_len+1:all_len+1], bins=100, range=[0,400], alpha=1, color='gold', edgecolor='gold', label="pt old reg", histtype='step', log=False)
            plt.hist(v_data[:,data_len:all_len], bins=100, range=[0,400], alpha=1, color='blue', edgecolor='blue', label="pt mc", histtype='step', log=False)
            plt.hist(v_data[:,0:1], bins=100, range=[0,400], alpha=1, color='green', edgecolor='green', label="pt", histtype='step', log=False)
            plt.legend(loc=1)
            plt.grid(True, which='both')
            plt.xlabel('pT [GeV]')
            plt.ylabel('# evts.')
            plt.savefig("png/plot_pTspectrum"+"_at_epoch"+str(epoch)+".png")
            plt.clf()
            
            
            arr1inds = v_data[:,data_len].argsort()
            print arr1inds.shape, "shape"
            sorted_v_data = v_data[arr1inds]
            #sorted_prediction = prediction[arr1inds[::-1]]
            sorted_prediction = prediction[arr1inds]
            print "sort"
            plt.hist2d(range(len(sorted_prediction)), sorted_prediction[:,0], bins=100, norm=LogNorm(), range=[[0,len(sorted_prediction)],[0,400]])
            cb=plt.colorbar()
            #plt.plot(sorted_prediction[:,0]) 
            plt.plot(sorted_v_data[:,data_len], linewidth=3.0, color="lime")                       
            plt.savefig("png/"+"bias_test"+"_at_epoch"+str(epoch)+".png")
            cb.remove()
            plt.clf()
            plt.hist2d(range(len(sorted_prediction)),sorted_prediction[:,0]-sorted_v_data[:,data_len], bins=100, norm=LogNorm())
            cb=plt.colorbar()                               
            plt.savefig("png/"+"bias_test2"+"_at_epoch"+str(epoch)+".png")
            cb.remove()
            plt.clf()
            
            #compare as a function of pt
            plt.hist2d(v_data[:,0],-v_data[:,data_len]+prediction[:,0],bins=100, norm=LogNorm(), range=[[0,200],[-100,100]])
            cb=plt.colorbar()
            plt.xlabel('pT [GeV]')
            plt.ylabel('regression error [GeV]')
            plt.savefig("png/"+"plot_corrRainbowLog_diff"+str(epoch)+".png")
            cb.remove()
            plt.clf()

        
wHistory_ = wHistory()
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
pltH_=plotHistory()

dropoutRate=0.1


Inputs=Input(shape=(data_len,))
x=  Dense(50, activation='relu',init='lecun_uniform')(Inputs)
###x = Dropout(dropoutRate)(x)
x=  Dense(50, activation='relu',init='lecun_uniform')(x)
###x = Dropout(dropoutRate)(x)
x=  Dense(20, activation='relu',init='lecun_uniform')(x)
###x = Dropout(dropoutRate)(x)
x=  Dense(20, activation='relu',init='lecun_uniform')(x)
###x = Dropout(dropoutRate)(x)
out1 = Dense(1,init='lecun_uniform')(x)
out2 = Dense(1,init='lecun_uniform')(x)
out3 = Dense(1,init='lecun_uniform')(x)
model = Model(input=Inputs, output=[out1,out2,out3])


#model = Sequential()
#model.add(Dense(50, input_dim=data_len, init='glorot_uniform', activation='relu'))
#model.add(Dropout(dropoutRate))
#model.add(Dense(50, init='lecun_uniform', activation='relu'))
#model.add(Dropout(dropoutRate))
#model.add(Dense(50, init='glorot_uniform', activation='relu'))
#model.add(Dropout(dropoutRate))
#model.add(Dense(20, init='glorot_uniform', activation='relu'))
#model.add(Dropout(dropoutRate))
#model.add(Dense(3, init='glorot_uniform'))
print'compiling'

print'compiling'


model.compile(loss=['mse',tilted_loss_1,tilted_loss_2], optimizer='adam',metrics=['mse', 'mae'])

model.summary()


print "fitting"

#history=model.fit(sc.transform(t_data[:,0:data_len]) , t_data[:,data_len:all_len], nb_epoch=50, verbose=1,batch_size=128, 
                  #callbacks=[wHistory_, reduce_lr], validation_data=(sc.transform(v_data[:,0:data_len]) , v_data[:,data_len:all_len]))
                  
history=model.fit(scaled_t_data , [t_data[:,data_len:all_len]/t_data[:,0:1],t_data[:,data_len:all_len]/t_data[:,0:1],t_data[:,data_len:all_len]/t_data[:,0:1]],
 nb_epoch=200, verbose=2,batch_size=5120,callbacks=[wHistory_, pltH_, reduce_lr], 
validation_data=(scaled_v_data , [v_data[:,data_len:all_len]/v_data[:,0:1],v_data[:,data_len:all_len]/v_data[:,0:1],v_data[:,data_len:all_len]/v_data[:,0:1]]))

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
 
