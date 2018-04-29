# import packages
from conf import myConfig as config
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.layers import Activation
from keras import optimizers
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import argparse
import numpy as np

# create CNN model
model=Sequential()
model.add(Conv2D(64,(3,3),padding="same",input_shape=(None,None,1)))
model.add(Activation('relu'))
for layers in range(2,16+1):
    model.add(Conv2D(64,(3,3),padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
model.add(Conv2D(1,(3,3),padding="same"))

# load the data and normalize it
cleanImages=np.load(config.data)
print(cleanImages.dtype)
cleanImages=cleanImages/255.0
cleanImages=cleanImages.astype('float32')

# define augmentor and custom flow
aug = ImageDataGenerator(rotation_range=30, fill_mode="nearest")
def myFlow(generator,X):
    for batch in generator.flow(x=X,batch_size=config.batch_size,seed=7):
        trueNoiseBatch=np.random.normal(0.0,config.sigma/255.0,batch.shape)
        noisyImagesBatch=batch+trueNoiseBatch
        yield (noisyImagesBatch,trueNoiseBatch)

# define custom learning rate scheduler
def lr_decay(epoch):
    initAlpha=0.001
    factor=0.5
    dropEvery=5
    alpha=initAlpha*(factor ** np.floor((1+epoch)/dropEvery))
    return float(alpha)
callbacks=[LearningRateScheduler(lr_decay)]


# define custom loss, compile the model
print("[INFO] compilingTheModel")
opt=optimizers.Adam(lr=0.001)
def custom_loss(y_true,y_pred):
    diff=y_true-y_pred
    res=K.sum(diff*diff)/(2*config.batch_size)
    return res
model.compile(loss=custom_loss,optimizer=opt)

# train
model.fit_generator(myFlow(aug,cleanImages),
epochs=config.epochs,steps_per_epoch=len(cleanImages)//config.batch_size,callbacks=callbacks,verbose=1)

# save the model
model.save('myModel.h5')
