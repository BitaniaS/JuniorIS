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
from keras import regularizers
from keras.regularizers import L2
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger



'''
# calculating accurary and F1 for each layer 
class LayerwiseAccuracyCallback(Callback):
    def __init__(self, model, x, y):
        super().__init__()
        self.model = model
        self.x = x
        self.y = y
    
    def on_epoch_end(self, epoch, logs=None):
        layer_accuracies = []
        for layer in self.model.layers:
            if hasattr(layer, 'output'):
                y_pred = layer.predict(self.x)
                y_pred = np.round(y_pred).astype(int)
                layer_acc = accuracy_score(self.y, y_pred)
                layer_accuracies.append(layer_acc)
            else:
                layer_accuracies.append(0)
        logs['layer_accuracies'] = layer_accuracies

class LayerwiseF1ScoreCallback(Callback):
    def __init__(self, model, x, y):
        super().__init__()
        self.model = model
        self.x = x
        self.y = y
    
    def on_epoch_end(self, epoch, logs=None):
        layer_f1_scores = []
        for layer in self.model.layers:
            if hasattr(layer, 'output'):
                y_pred = layer.predict(self.x)
                y_pred = np.round(y_pred).astype(int)
                layer_f1 = f1_score(self.y, y_pred)
                layer_f1_scores.append(layer_f1)
            else:
                layer_f1_scores.append(0)
        logs['layer_f1_scores'] = layer_f1_scores
'''


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

    seg_labels=np.zeros((320,320,1)) # creating a Numpy Array width * height*channels 
    img=cv2.resize(img,(320,320)) #resize img to height and width of 512 (same case as the gray scale thingy, do I need to preprocess ?)
    
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

# img=cv2.imread(f'./PageSegData/PageImg/229.JPG',0)
# cv2.imshow('image',img)

# img2=cv2.imread(f'./PageSegData/PageImg/229.JPG',0)
# visualize(img,img2)
# # cv2.imshow('image',img)
# # imga = cv2.imread('./PageSegData/PageSeg/1_mask.png')
# # get_segmented_img(imga)


def preprocess_img(img):
    img=cv2.resize(img,(320,320))
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


# batch generator 

def batch_generator(filelist,n_classes,batch_size,augment):
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
      img=cv2.imread(f'./PageSegData/PageImg/{fn}.JPG',0)
      ret,img=cv2.threshold(img,150,255,cv2.THRESH_BINARY_INV)
      img=cv2.resize(img,(320,320))
      img=np.expand_dims(img,axis=-1)
      img=img/255

      seg=cv2.imread(f'./PageSegData/PageSeg/{fn}_mask.png',1)
      seg=get_segmented_img(seg,n_classes)

      if augment:
          seed = np.random.randint(1,100000)
          img = datagen.random_transform(img, seed=seed)
          seg = datagen.random_transform(seg, seed=seed)

      X.append(img)
      Y.append(seg)
    yield np.array(X),np.array(Y)



def plot_epochMetric(history,metric):
       
       train_metrics = history.history[metric]
       val_metrics = history.history['val_'+metric]
       epochs = range(1, len(train_metrics) + 1)
       plt.plot(epochs, train_metrics)
       plt.plot(epochs, val_metrics)
       plt.title('Training and validation '+ metric)
       plt.xlabel("Epochs")
       plt.ylabel(metric)
       plt.legend(["train_"+metric, 'val_'+metric])
       plt.show()

metric =  ['accuracy']
input_size = (320,320,1)


params = {
    'lr': [1e-3, 1e-4, 1e-5], 
    'batch_size': [4,8,16],
    'epochs': [3, 5, 7],
    'filters': [8, 16, 32],
    'steps':[100,150,200],
    'w_decay' : [0.00001, 0.0001, 0.001],
    'step' :[100,150,200],
    'augment_v': ['True','False']
}

val_loss_list = []
val_acc_list = []
# f1_list = []
# prec_list = []
# rec_list = []

#splitting the dataset (lela type of splitting new metekem yalebign)
random.shuffle(image_list)
file_train=image_list[0:int(0.70*len(image_list))]
file_valid=image_list[int(0.70*len(image_list)): int(0.90*len(image_list))]
file_test=image_list[int(0.90*len(image_list)):]

# #data format is channels last in this case. 
def unet(n_items):

    for i in range (n_items):
        valid_gen = batch_generator(file_valid, 2,20,augment=False)
        # Randomly choose hyperparameters
        learn_r = random.choice(params['lr'])
        batch_s = random.choice(params['batch_size']) 
        epochs = random.choice(params['epochs'])
        filters = random.choice(params['filters'])
        w_decay = random.choice(params['w_decay'])   
        step = random.choice(params['step']) 
        augment_v = random.choice(params['augment_v']) 


        inputs = Input(input_size)
        conv1 = Conv2D(filters, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(filters, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(filters, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(filters, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(filters, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(filters, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        up4 = UpSampling2D(size=(2, 2))(pool3)
        up4 = Conv2D(filters, 2, activation='relu', padding='same')(up4)
        up4 = Concatenate()([up4, conv3])
        up4 = Conv2D(filters, 3, activation='relu', padding='same')(up4)

        up5 = UpSampling2D(size=(2, 2))(up4)
        up5 = Conv2D(filters, 2, activation='relu', padding='same')(up5)
        up5 = Concatenate()([up5, conv2])
        up5 = Conv2D(filters, 3, activation='relu', padding='same')(up5)

        up6 = UpSampling2D(size=(2, 2))(up5)
        up6 = Conv2D(filters, 2, activation='relu', padding='same')(up6)
        up6 = Concatenate()([up6, conv1])
        up6 = Conv2D(filters, 3, activation='relu', padding='same')(up6)

        outputs = Conv2D(filters, 1, activation='softmax')(up6)
        model = Model(inputs=[inputs], outputs=[outputs])

        
        # the order of it mastekakel alebign
        mc = ModelCheckpoint('best_weight{i}_epoch{epoch:03d}.h5', monitor=' precision, recall',
                             save_best_only=True)
        model.compile(optimizer = Adam(learning_rate= learn_r), loss = 'binary_crossentropy', metrics= metric
                     )
     
        # csv_logger = CSVLogger('training_history.csv',append = True)
        history = model.fit(batch_generator(file_train,2,batch_s,augment = augment_v),epochs=epochs,steps_per_epoch=step,validation_data=valid_gen,
                   validation_steps=step,callbacks=[mc],shuffle=1)
       
        val_loss_list.append(history.history['loss'])
        val_acc_list.append(history.history['accuracy'])
        plot_epochMetric(history,'accuracy')
        plot_epochMetric(history,'loss')
                         
        # f1_list.append(history.history['f1_score'])
        # prec_list.append(history.history['precision'])
        # rec_list.append(history.history['recall'])
        model.summary()
  
        

        # score = model.evaluate(valid_gen, metrics = metric ) #the hyperparameters mastekakel alebign

    #model.summary()
    # if(pretrained_weights):
    #     model.load_weights(pretrained_weights)
        return model
       
    #visualizing the metrics

def plot_rsearch():

    plt.plot(val_loss_list, label='val_loss')
    plt.plot(val_acc_list, label='accuracy')
    # plt.plot(f1_list, label='f1_score')
    # plt.plot(prec_list, label='precision')
    # plt.plot(rec_list, label='recall')
    plt.legend()
    plt.show()


   
unet(4)
# model=unet()
# model.summary() 





# #TRAINING THE MODEL
# # this saves the only the weigths of the model after every epoch 
# # I am not sure if i should also use the SAVE_BEST_ONLY parameter 
# mc = ModelCheckpoint('weights{epoch:08d}.h5', 
#                                      save_weights_only=True, period=1)
# # callback_accuracy = LayerwiseAccuracyCallback(model, x_val, y_val)
# # callback_f1_score = LayerwiseF1ScoreCallback(model, x_val, y_val)

# # is should change the steps_per_epoch 
# # model.fit_generator(batch_generator(file_train,2,2),epochs=3,steps_per_epoch=1000,validation_data=batch_generator(file_valid,2,2),
# #                     validation_steps=400,callbacks=[mc],shuffle=1)




# def random_search(params, n_iter=10):
#     best_score = 0
#     best_params = None
#     for i in range(n_iter):
#         # Randomly choose hyperparameters
#         lr = random.choice(params['lr'])
#         batch_size = random.choice(params['batch_size']) 
#         epochs = random.choice(params['epochs'])
#         filters = random.choice(params['filters'])
        
#         # Build model with chosen hyperparameters
#         model = unet(lr=lr, batch_size=batch_size, epochs=epochs, filters=filters)
        
#         # Train model and get validation score
        
#         # metric =  ['accuracy','f1_macro','precision','recall']
#         model.fit(batch_generator(file_train,2,batch_size = batch_size),epochs=3,steps_per_epoch=1000,validation_data=valid_gen,
#                     validation_steps=400,callbacks=[mc],shuffle=1)
        
#         # model.fit_generator(train_gen, epochs=epochs, validation_data=val_gen)
#         score = model.evaluate(valid_gen, metrics = metric )
        
#         # Save best model
#         if score > best_score:
#             best_score = score
#             best_params = {
#                 'lr': lr,
#                 'batch_size': batch_size, 
#                 'epochs': epochs,
#                 'filters': filters
#             }
            
#     print(f'Best score: {best_score}') 
#     print(f'Best hyperparameters: {best_params}')


