#importLibraries
from keras.models import load_model
from keras.models import Model
from conf import myConfig as config
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
import cv2
import numpy as np
from skimage.measure import compare_psnr
import argparse
from pathlib import Path
import keras.backend as K

#ParsingArguments
parser=argparse.ArgumentParser()
parser.add_argument('--dataPath',dest='dataPath',type=str,default='./Set12/',help='testDataPath')
parser.add_argument('--weightsPath',dest='weightsPath',type=str,default='./myModel.h5',help='pathOfTrainedCNN')
args=parser.parse_args()

#createModel, loadWeights
def custom_loss(y_true,y_pred): #this is required for loading a keras-model created with custom-loss
    diff=y_true-y_pred
    res=K.sum(diff*diff)/(2*config.batch_size)
    return res
nmodel=load_model(args.weightsPath,custom_objects={'custom_loss':custom_loss})
print('nmodel is loaded')

#createArrayOfTestImages
p=Path(args.dataPath)
listPaths=list(p.glob('./*.png'))
imgTestArray = []
for path in listPaths:
    imgTestArray.append(((cv2.resize
    (cv2.imread(str(path),0),(200,200),
    interpolation=cv2.INTER_CUBIC))))
imgTestArray=np.array(imgTestArray)/255
    
#calculatePSNR
sumPSNR=0
for i in range(0,len(imgTestArray)):
    cv2.imshow('trueCleanImage',imgTestArray[i]); cv2.waitKey(0)
    noisyImage=imgTestArray[i]+np.random.normal(0.0,25/255, 
        imgTestArray[i].shape)
    cv2.imshow('noisyImage',noisyImage); cv2.waitKey(0)
        #print(np.expand_dims(np.expand_dims(noisyImage,axis=2),axis=0).shape)
    error=nmodel.predict(np.expand_dims(np.expand_dims(noisyImage,axis=2),axis=0))
    predClean=noisyImage-np.squeeze(error)
        #print(error.min(),error.max())
    cv2.imshow('predCleanImage',predClean); cv2.waitKey(0)
    psnr=compare_psnr(imgTestArray[i],predClean)
    sumPSNR=sumPSNR+psnr
    cv2.destroyAllWindows()
avgPSNR=sumPSNR/len(imgTestArray)
print('avgPSNR on test-data',avgPSNR)
print(sumPSNR)

