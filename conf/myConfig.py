#import packages
import os

epochs=50
batch_size=128

#noise-level for training
sigma=25.0

#path to training data
data='./trainingPatch/img_clean_pats.npy'

#path to generate the data
genDataPath='./genData/'

#path to save the genPatches
save_dir='./trainingPatch/'

pat_size=40

stride=10

step=0

