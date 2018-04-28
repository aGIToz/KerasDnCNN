#command: python testPSNR.py --dataPath /home/amitoz/Documents/MasterProjet/code/test/data/test/Set12/*.png --weightsPath /home/amitoz/Documents/MasterProject/code/BetterModel/dncnn2.model

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
parser.add_argument('--dataPath',dest='dataPath',type=str,default='.',help='testDataPath')
parser.add_argument('--weightsPath',dest='weightsPath',type=str,default='.',help='pathOfTrainedCNN')
args=parser.parse_args()

#createModel, loadWeights
def custom_loss(y_true,y_pred):
    diff=y_true-y_pred
    res=K.sum(diff*diff)/(2*config.batch_size)
    return res

nmodel=load_model(args.weightsPath,custom_objects={'custom_loss':custom_loss})
print('nmodel is loaded')

#createArrayOfTestImages
print(args.dataPath)
p=Path(args.dataPath)
listPaths=list(p.glob('./*.png'))
imgTestArray = []
for path in listPaths:
    imgTestArray.append(((cv2.resize
    (cv2.imread(str(path),0),(200,200),
    interpolation=cv2.INTER_CUBIC))))
print(imgTestArray[1].max())
imgTestArray=np.array(imgTestArray)/255
print('lenImages',len(imgTestArray))
print(imgTestArray.shape)

iimgTestArray=np.load('../img_clean_pats.npy')
iimgTestArray=imgTestArray/255
iimgTestArray=np.squeeze(imgTestArray)
print('iimgTestArray Dim:',iimgTestArray.shape)

#calculatePSNR
sumPSNR=0
for i in range(0,len(imgTestArray)):
    cv2.imshow('trueCleanImage',imgTestArray[i]); cv2.waitKey(0)
    noisyImage=imgTestArray[i]+np.random.normal(0.0,25/255, 
        imgTestArray[i].shape)
    print("Orignal",imgTestArray[i])
    print(imgTestArray[i].min(),imgTestArray[i].max())
    print(imgTestArray[i].mean())
    print("Noise",noisyImage)
    print(noisyImage.min(),noisyImage.max())
    print(noisyImage.mean())
    cv2.imshow('noisyImage',noisyImage); cv2.waitKey(0)
    print(np.expand_dims(np.expand_dims(noisyImage,axis=2),axis=0).shape)
    error=nmodel.predict(np.expand_dims(np.expand_dims(noisyImage,axis=2),axis=0))
    predClean=noisyImage-np.squeeze(error)
    print("ErrorIm",error)
    print(error.min(),error.max())
    print(error.mean())
    print("PreImage",predClean)
    print(predClean.min(),predClean.max())
    print(predClean.mean())
    cv2.imshow('predCleanImage',predClean); cv2.waitKey(0)
    psnr=compare_psnr(imgTestArray[i],predClean)
    sumPSNR=sumPSNR+psnr
    cv2.destroyAllWindows()
avgPSNR=sumPSNR/len(imgTestArray)
print('avgPSNR',avgPSNR)
print(sumPSNR)

