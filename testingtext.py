# #Importing required libraries 

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
import random
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint




'''
this code reads the file names of images in the /content/PageSegData/PageImg/' directory
stores them in a list called image_list. 
It then removes the file extension from each filename and then
stores the result in the same image_list.
os.listdir : gets a list of all files and directories in the specified directory
'''
image_list=os.listdir('./PageSegData/PageImg')
image_list=[filename.split(".")[0]for filename in image_list] #use os.path.splittext() as an alternate 


'''
This function takes two images as input, displays them side by side, 
and titles them "Image" and "Segmented Image". 
This function is used later in the script to display images.
'''
def visualize(img,seg_img):
    """
    Visualizes image
    """
    plt.figure(figsize=(20,20))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title('Image')
    plt.subplot(1,2,2)
    plt.imshow(seg_img,cmap='gray')
    plt.title('Segmented Image')
    plt.show()

'''
This creates a segmentation mask where each pixel in the image is assigned a label or category. 
In this case, all the pixels are initially labeled as 0, indicating that they do not belong to any category.
'''

#to convert the segmentation mask into one-hot encoded representation 
def get_segmented_img(img,n_classes): # also known as binary segmentation mask 

    seg_labels=np.zeros((512,512,1)) # creating a Numpy Array width * height*channels 
    img=cv2.resize(img,(512,512)) #resize img to height and width of 512 (same case as the gray scale thingy, do I need to preprocess ?)
    
    img=img[:,:,0] # i'm not sure if this line converts it to grey scale or not ( do I need to do the preprocessing part ? )
     # selecting only the first channel of the resized image array this effectively converts it to grayscale 
    # Image preprocessing needs to happen to convert img to gray scale
    # The preprocessing function can perform the following (color space conversion, filtering and etc). 
    # Look into the function I already have and also the research paper which talks about the best methods for preprocessing images 


    cl_list=[0,24] #NOT SURE IF I NEED THIS

    
    seg_labels[:,:,0]=(img!=0).astype(int) # if the pixels in img are not 0 then set the corresponding pixel in 
    # seg_label to 1 so all non zero pixel in img have a mask of 1 and all zero pixel in img have a mask of 0
    
    # cv2.imshow('Segmentation Mask', seg_labels)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return seg_labels

# imga = cv2.imread('./PageSegData/PageSeg/1_mask.png')
# get_segmented_img(imga)


def preprocess_img(img):
    img=cv2.resize(img,(512,512))
    return img

#function that creates a numpy array of input images (actual text images)
# and output segmentation masks the output masks we have from the get_segmented_img function
# def batch_generator(filelist,n_classes,batch_size):
#   while True:
#     X=[]
#     Y=[]
#     for i in range(batch_size):
#       fn=random.choice(filelist)
#       img=cv2.imread(f'/PageSegData/PageImg/{fn}.JPG',0)
#       ret,img=cv2.threshold(img,150,255,cv2.THRESH_BINARY_INV)
#       img=cv2.resize(img,(512,512))
#       img=np.expand_dims(img,axis=-1)
#       img=img/255

#       seg=cv2.imread(f'/PageSegData/PageSeg/{fn}_mask.png',1)
#       seg=get_segmented_img(seg,n_classes)

#       X.append(img)
#       Y.append(seg)
#     yield np.array(X),np.array(Y)


#batch generator 

def batch_generator(filelist,n_classes,batch_size,augment=True):
  datagen = ImageDataGenerator(
      rotation_range=30,
      width_shift_range=0.1, #shifts the width of all the pixels of the image keza teg lay yalew to the beginning part yehedal
      height_shift_range=0.1, #distorts image a bit by shifting some pixels along a given axis
      shear_range=0.1,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='constant',
      cval=0.0/255.0,
    #means that the constant value used for padding or filling empty pixels 
    #in the image will be black, and that the value will be scaled to the 
    # range between 0 and 1 for compatibility with certain deep learning frameworks.
      preprocessing_function=None)

  while True:
    X=[]
    Y=[]
    for i in range(batch_size):
      fn=random.choice(filelist)
      img=cv2.imread(f'/content/PageSegData/PageImg/{fn}.JPG',0)
      ret,img=cv2.threshold(img,150,255,cv2.THRESH_BINARY_INV)
      img=cv2.resize(img,(512,512))
      img=np.expand_dims(img,axis=-1)
      img=img/255

      seg=cv2.imread(f'/content/PageSegData/PageSeg/{fn}_mask.png',1)
      seg=get_segmented_img(seg,n_classes)

      if augment:
          seed = np.random.randint(1,100000)
          img = datagen.random_transform(img, seed=seed)
          seg = datagen.random_transform(seg, seed=seed)

      X.append(img)
      Y.append(seg)

    yield np.array(X),np.array(Y)





# #data format is channels last in this case. 
def unet(pretrained_weights = None,input_size = (512,512,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    #decoder network mijemerebet
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    model = Model(inputs,conv10)


    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

model=unet()
model.summary() 

#splitting the dataset (lela type of splitting new metekem yalebign)
random.shuffle(image_list)
file_train=image_list[0:int(0.70*len(image_list))]
file_valid=image_list[int(0.70*len(image_list)): int(0.90*len(image_list))]
file_test=image_list[int(0.90*len(image_list)):]

#TRAINING THE MODEL
# this saves the only the weigths of the model after every epoch 
# I am not sure if i should also use the SAVE_BEST_ONLY parameter 
mc = ModelCheckpoint('weights{epoch:08d}.h5', 
                                     save_weights_only=True, period=1)

# is should change the steps_per_epoch 
model.fit_generator(batch_generator(file_train,2,2),epochs=3,steps_per_epoch=1000,validation_data=batch_generator(file_test,2,2),
                    validation_steps=400,callbacks=[mc],shuffle=1)



