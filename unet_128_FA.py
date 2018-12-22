# -*- coding: utf-8 -*-
import sys

import os
import numpy as np
from keras.models import Model
from keras.layers import Dropout
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
import keras.utils
import keras.optimizers
import keras.preprocessing.image
from math import floor, ceil
from skimage import measure
import postprocess

def Activation(name):
    if name == 'prelu':
        return keras.layers.advanced_activations.PReLU()
    elif name == 'lrelu':
        return keras.layers.advanced_activations.LeakyReLU()
    elif name == 'elu':
        return keras.layers.advanced_activations.ELU()
    else:
        return keras.layers.Activation(name)

activation_layer = 'relu'
kernel_size = 3
def contract(prev_layer, n_kernel,pool_size=2):
    conv = Conv2D(n_kernel, kernel_size, padding='same', use_bias=True)(prev_layer)
    conv = Activation(activation_layer)(conv)
    n_kernel = n_kernel << 1
    conv = Conv2D(n_kernel, kernel_size, padding='same', use_bias=False)((conv))
    conv = Activation(activation_layer)(BatchNormalization()(conv))
    pool = MaxPooling2D(pool_size=pool_size,strides=pool_size)((conv))

    return conv, pool

def expand(prev_layer, left_layer, n_kernel,pool_size=2):
    dx,dy = (left_layer.shape[1].value-prev_layer.shape[1].value*pool_size)/2, (left_layer.shape[2].value-prev_layer.shape[2].value*pool_size)/2
    crop_size = ((floor(dx),ceil(dx)),(floor(dy),ceil(dy)))
    up = Concatenate(axis=3)([UpSampling2D(size=pool_size)(prev_layer), Cropping2D(crop_size)(left_layer)])
    conv = Conv2D(n_kernel, kernel_size, padding='same', use_bias=True)(up)
    conv = Activation(activation_layer)(conv)
    n_kernel = n_kernel >> 1
    conv = Conv2D(n_kernel, kernel_size, padding='same', use_bias=False)((conv))
    conv = Activation(activation_layer)(BatchNormalization()(conv))
    return conv

def create_model(input_shape, n_output_classes):
    inputs = Input(input_shape)
    n_kernel = 32
    conv1, pool1 = contract(inputs, n_kernel)
    n_kernel = conv1.shape[-1].value
    conv2, pool2 = contract(pool1, n_kernel)
    n_kernel = conv2.shape[-1].value
    conv3, pool3 = contract(pool2, n_kernel)
    n_kernel = conv3.shape[-1].value
    conv4, pool4 = contract(pool3, n_kernel)
    n_kernel = conv4.shape[-1].value
    conv5, pool5 = contract(pool4, n_kernel)
    n_kernel = conv5.shape[-1].value

    conv = expand(conv5, conv4, n_kernel)
    n_kernel = conv.shape[-1].value
    conv = expand(conv, conv3, n_kernel)
    n_kernel = conv.shape[-1].value
    conv = expand(conv, conv2, n_kernel)
    n_kernel = conv.shape[-1].value
    conv = expand(conv, conv1, n_kernel)

    output = Conv2D(n_output_classes, 1, activation='sigmoid', padding='same')(conv)

    return Model(inputs=inputs, outputs=output)

def JaccardIndex(a,b):
    return np.sum(a&b)/np.sum(a|b)

def DiceCoeff(a,b):
    return 2*np.sum(a&b)/(np.sum(a)+np.sum(b))

#  Surface Distance Based Methods
from scipy.ndimage import morphology
import sklearn.neighbors
def average_symmetric_surface_distance(dist_a, dist_b):
    return (np.sum(dist_a)+np.sum(dist_b))/(dist_a.size+dist_b.size)
def ASD(dist_a, dist_b):
    ''' Alias of :func:`average_symmetric_surface_distance`'''
    return average_symmetric_surface_distance(dist_a, dist_b)
def extract_contour(mask, connectivity=None, keep_border=False):
    if connectivity is None:
        connectivity = mask.ndim
    conn = morphology.generate_binary_structure(mask.ndim, connectivity)
    return np.bitwise_xor(mask, morphology.binary_erosion(mask, conn, border_value=0 if keep_border else 1))
def calculate_surface_distances(a, b, spacing=None, connectivity=None, keep_border=False):
    '''Extract border voxels of a and b, and return both a->b and b->a surface distances'''
    if spacing is None:
        spacing = np.ones(a.ndim)
    pts_a = np.column_stack(np.where(extract_contour(a,connectivity,keep_border))) * np.array(spacing)
    pts_b = np.column_stack(np.where(extract_contour(b,connectivity,keep_border))) * np.array(spacing)
    if not pts_a.any() or not pts_b.any():
        return np.array([float('nan'),float('nan')])
    tree = sklearn.neighbors.KDTree(pts_b)
    dist_a2b,_ = tree.query(pts_a)
    tree = sklearn.neighbors.KDTree(pts_a)
    dist_b2a,_ = tree.query(pts_b)
    return dist_a2b, dist_b2a

def evaluate(label1,label2,spacing):
    max_label = max(np.max(label1),np.max(label2))
    JIs,DCs,ASDs = [],[],[]
    for i in range(1,max_label+1):
        a = label1 == i
        b = label2 == i
        JIs.append(JaccardIndex(a,b))
        DCs.append(DiceCoeff(a,b))
        dist_a2b,dist_b2a = calculate_surface_distances(a, b, spacing, connectivity=None, keep_border=None)
        ASDs.append(ASD(dist_a2b,dist_b2a))
    return JIs,DCs,ASDs

def largest_CC(image, n=1):
    labels = measure.label(image, connectivity=3, background=0)
    area = np.bincount(labels.flat)
    if (len(area)>1):
        return labels == (np.argmax(area[1:])+1)
    else:
        return np.zeros(labels.shape,np.bool)

def refine_labels(labels):
    return labels
    refined = np.zeros_like(labels)
    for i in range(1,np.max(labels)+1):
        cc = largest_CC(labels==i)
        refined[cc] = i
    return refined

import json, pickle
def save_object(obj, filename):
    if os.path.splitext(filename)[1] == '.json':
        with open(filename,'w') as f:
            json.dump(obj,f,indent=2)
    else:
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)
import mhd
import tqdm
import glob

datadir = r"/win/salmon/user/fatemeh/Unet_test1_masseter/Unet_masseter_dataset"

dataset = {}
spacings = {}
#IDs = [os.path.basename(r)[0:5] for r in glob.glob(os.path.join(datadir,'k*_image.mhd'))]
for ID in tqdm.tqdm(os.listdir(datadir)):
    print(ID)
#    original,h = mhd.read(os.path.join(datadir,'{}_image.mhd'.format(ID)))
    original,h = mhd.read(os.path.join(datadir,ID,'original.mhd'))
    label = mhd.read(os.path.join(datadir,ID,'label.mha'))[0]
    #label = mhd.read(os.path.join(datadir,'{}_label_muscle.mhd'.format(ID)))[0]
#    label = mhd.read(os.path.join(datadir,'{}_label_skin.mhd'.format(ID)))[0]
    spacings[ID] = h['ElementSpacing'][::-1] # reversing is required to make it [z y x] order
    data = {}
    data['x'] = np.expand_dims((original/255.0).astype(np.float32),-1)
    data['y'] = np.expand_dims(label,-1)
    dataset[ID] = data
x_shape = next(iter(dataset.values()))['x'].shape[1:]
n_classes = np.max(label) + 1

from datetime import datetime
result_basedir = 'unet_train_' + datetime.today().strftime("%y%m%d_%H%M%S")
os.makedirs(result_basedir,exist_ok=True)
import sys,shutil
shutil.copyfile(sys.argv[0],os.path.join(result_basedir,os.path.basename(sys.argv[0]))) #copy this script

all_IDs = sorted(dataset.keys())

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

import tensorflow as tf        
class TB(keras.callbacks.TensorBoard):
    def __init__(self, log_every=1, **kwargs):
        super().__init__(**kwargs)
        self.log_every = log_every
        self.counter = 0
    
    def on_batch_end(self, batch, logs=None):
        self.counter+=1
        if self.counter%self.log_every==0:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.counter)
            self.writer.flush()
        
        super().on_batch_end(batch, logs)
        
from keras.utils.vis_utils import model_to_dot
#from keras.utils import multi_gpu_model
#import pydot_ng
#import postprocess
k_fold = 4
for exp_no in range(1):
    groups = np.split(np.random.permutation(np.array(all_IDs,dtype=str)),k_fold)
    exp_dir = os.path.join(result_basedir,'exp{0}'.format(exp_no))
    os.makedirs(exp_dir, exist_ok=True)
    save_object([g.tolist() for g in groups],os.path.join(exp_dir,'groups.json'))
    print(groups)
    JIs,DCs,ASDs = {},{},{}
    refined_JIs,refined_DCs,refined_ASDs = {},{},{}
    for group_no,test_IDs in enumerate(groups):
        result_dir = os.path.join(exp_dir,'g{0}'.format(group_no))
        os.makedirs(result_dir, exist_ok=True)
        model = create_model(x_shape,n_classes)
        gpu_count = 2
#        parallel_model = multi_gpu_model(model, gpus=gpu_count)
        opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-2)
#        parallel_model.compile(opt, 'sparse_categorical_crossentropy')
        model.compile(opt, 'sparse_categorical_crossentropy')
        
        #svg_filename = os.path.join(result_dir, 'network_structure.svg')
        #d = model_to_dot(model, show_shapes=True, show_layer_names=False)#, to_file=svg_filename)

        es = keras.callbacks.EarlyStopping(monitor='loss', patience=1, verbose=1, mode='auto')
#        tb = keras.callbacks.TensorBoard(batch_size=2, histogram_freq=0)
        tb = TB(log_dir="./logs3")
        history = LossHistory()
        train_IDs = [ID for ID in dataset.keys() if ID not in test_IDs]
        x_train = np.concatenate([dataset[ID]['x'] for ID in train_IDs])
        y_train = np.concatenate([dataset[ID]['y'] for ID in train_IDs])
        
        x_test = np.concatenate([dataset[ID]['x'] for ID in test_IDs])
        y_test = np.concatenate([dataset[ID]['y'] for ID in test_IDs])
        print('test',test_IDs.tolist())
        print('train',train_IDs)
        epochs = 6
#        callbacks=[es,tb,history]
        callbacks=[es,history]
#        model.fit(x_train, y_train, batch_size=2, epochs=epochs, callbacks=callbacks, validation_data=(x_test,y_test))
#        parallel_model.fit(x_train, y_train, batch_size=2, epochs=epochs, callbacks=callbacks)
        model.fit(x_train, y_train, batch_size=2, epochs=epochs, callbacks=callbacks)
        
        #save loss history
        np.savetxt(os.path.join(result_dir, 'training.log'), history.losses, delimiter=",", fmt='%g')
        
        #save model
        with open(os.path.join(result_dir, 'unet_model.json'),'w') as f:
            f.write(model.to_json())
        model.save_weights(os.path.join(result_dir, 'unet_model_weights.h5'))
            
        for test_ID in test_IDs:
            x_test = dataset[test_ID]['x']
            predict_y = model.predict(x_test,batch_size=2,verbose=True)
            
            predict_label = np.argmax(predict_y,axis=3).astype(np.uint8)
            mhd.write(os.path.join(result_dir,test_ID+'.mhd'),np.squeeze(predict_label.astype(np.uint8)),header={'ElementSpacing':spacings[test_ID][::-1]})
            
            JIs[test_ID],DCs[test_ID],ASDs[test_ID] = evaluate(predict_label,np.squeeze(dataset[test_ID]['y']),spacings[test_ID])
            refined = postprocess.refine_label(predict_label)
            mhd.write(os.path.join(result_dir,'refined_'+test_ID+'.mhd'),np.squeeze(refined.astype(np.uint8)),header={'ElementSpacing':spacings[test_ID][::-1]})
            refined_JIs[test_ID],refined_DCs[test_ID],refined_ASDs[test_ID] = evaluate(refined,np.squeeze(dataset[test_ID]['y']),spacings[test_ID])
            
    np.savetxt(os.path.join(exp_dir,'JI.csv'), np.stack([JIs[ID] for ID in all_IDs]), delimiter=",",fmt='%g')
    np.savetxt(os.path.join(exp_dir,'Dice.csv'), np.stack([DCs[ID] for ID in all_IDs]), delimiter=",",fmt='%g')
    np.savetxt(os.path.join(exp_dir,'ASD.csv'), np.stack([ASDs[ID] for ID in all_IDs]), delimiter=",",fmt='%g')
    np.savetxt(os.path.join(exp_dir,'refined_JI.csv'), np.stack([refined_JIs[ID] for ID in all_IDs]), delimiter=",",fmt='%g')
    np.savetxt(os.path.join(exp_dir,'refined_Dice.csv'), np.stack([refined_DCs[ID] for ID in all_IDs]), delimiter=",",fmt='%g')
    np.savetxt(os.path.join(exp_dir,'refined_ASD.csv'), np.stack([refined_ASDs[ID] for ID in all_IDs]), delimiter=",",fmt='%g')
