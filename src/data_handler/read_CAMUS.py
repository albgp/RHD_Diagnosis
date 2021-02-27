from eisen.datasets import CAMUS
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def readImg2CH(instance):
    img = io.imread(instance["image_2CH"], plugin='simpleitk') 

def loadCamus2CH(location):
    dset_train = CAMUS(location+'training/',True,with_4CH=False)
    dset_test = CAMUS(location+'testing/',True,with_4CH=False)
    trainX=[]
    trainy=[]
    testX=[]

    for instance in dset_train:
        img = io.imread(location+'training/'+instance["image_2CH"], plugin='simpleitk') 
        img_label = io.imread(location+'training/'+instance["label_2CH"], plugin='simpleitk') 
        trainX.append(img[0,:,:])
        trainy.append(img_label[0,:,:])
    
    for instance in dset_test:
        img = io.imread(location+'testing/'+instance["image_2CH"], plugin='simpleitk') 
        testX.append(img[0,:,:])
    
    return trainX, trainy, testX

def readImg4CH(instance):
    img = io.imread(instance["image_4CH"], plugin='simpleitk') 

def loadCamus4CH(location):
    dset_train = CAMUS(location+'training/',True,with_2CH=False)
    dset_test = CAMUS(location+'testing/',True,with_2CH=False)
    trainX=[]
    trainy=[]
    testX=[]

    for instance in dset_train:
        img = io.imread(location+'training/'+instance["image_4CH"], plugin='simpleitk') 
        img_label = io.imread(location+'training/'+instance["label_4CH"], plugin='simpleitk') 
        trainX.append(img[0,:,:])
        trainy.append(img_label[0,:,:])
    
    for instance in dset_test:
        img = io.imread(location+'testing/'+instance["image_4CH"], plugin='simpleitk') 
        testX.append(img[0,:,:])
    
    return trainX, trainy, testX