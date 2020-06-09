# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:31:15 2020

@author: Kather
"""
# Using flask to make an api 
# import necessary libraries and functions 
#import keras.backend.tensorflow_backend as tb
#tb._SYMBOLIC_SCOPE.value = True
from flask import Flask, jsonify, request 
import os
import cv2
from skimage.transform import rotate
from deskew import determine_skew
import numpy as np
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers.merge import average, concatenate
from keras.models import Model
from keras.layers.recurrent import LSTM
import random
import itertools
#import tensorflow as tf

# creating a Flask app 
app = Flask(__name__) 

os.chdir('C:/Users/Kather/Desktop/fahim/ML project/sent by surya/CRNN')
width = 128
height = 32
unicodes = list(np.load('unicodes.npy',allow_pickle=True))

def predict(img2,model):
    img2= preprocess(img2,(width,height),False)
    img2 = np.reshape(img2,(1,img2.shape[0],img2.shape[1],1))
    out = model.predict(img2)
    pred = decode(out)
    for word in pred:
        return str(word)
    
def ctcLambdaFunc(args):
    yPred, labels, inputLength, labelLength = args
    yPred = yPred[:,2:,:]
    loss = K.ctc_batch_cost(labels,yPred,inputLength,labelLength)
    return loss
    
def getModel(training):
    inputShape = (128,32,1)
    kernelVals = [5,5,3,3,3]
    convFilters = [32,64,128,128,256]
    strideVals = [(2,2),(2,2),(1,2),(1,2),(1,2)]
    rnnUnits = 256
    maxStringLen = 32
    inputs = Input(name = 'inputX', shape = inputShape, dtype = 'float32')
    inner = inputs
    for i in range(len(kernelVals)):
        inner = Conv2D(convFilters[i],(kernelVals[i],kernelVals[i]),padding = 'same',\
                       name = 'conv'+str(i), kernel_initializer = 'he_normal')(inner)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size = strideVals[i],name = 'max' + str(i+1))(inner)
    inner = Reshape(target_shape = (maxStringLen,rnnUnits), name = 'reshape')(inner)
    
    LSF = LSTM(rnnUnits,return_sequences=True,kernel_initializer='he_normal',name='LSTM1F')(inner)
    LSB = LSTM(rnnUnits,return_sequences=True, go_backwards = True, kernel_initializer='he_normal',name='LSTM1B')(inner)
    LSB = Lambda(lambda inputTensor: K.reverse(inputTensor,axes=1))(LSB)
    
    LS1 = average([LSF,LSB])
    LS1 = BatchNormalization()(LS1)
    
    LSF = LSTM(rnnUnits,return_sequences=True,kernel_initializer='he_normal',name='LSTM2F')(LS1)
    LSB = LSTM(rnnUnits,return_sequences=True, go_backwards = True, kernel_initializer='he_normal',name='LSTM2B')(LS1)
    LSB = Lambda(lambda inputTensor: K.reverse(inputTensor,axes=1))(LSB)
    
    LS2 = concatenate([LSF,LSB])
    LS2 = BatchNormalization()(LS2)
    yPred = Dense(len(unicodes)+1,kernel_initializer='he_normal',name='dense2')(LS2)
    yPred = Activation('softmax',name='softmax')(yPred)
    #Model(inputs = inputs,outputs = yPred).summary()
    
    labels = Input(name='label', shape=[32], dtype='float32')
    inputLength = Input(name='inputLen', shape=[1], dtype='int64')     # (None, 1)
    labelLength = Input(name='labelLen', shape=[1], dtype='int64')
    
    lossOut = Lambda(ctcLambdaFunc, output_shape=(1,), name='ctc')([yPred, labels, inputLength, labelLength])
    
    if training:
        return Model(inputs = [inputs, labels, inputLength, labelLength], outputs=[lossOut,yPred])
    return Model(inputs=[inputs], outputs=yPred) 

def labelsToText(labels):
    ret = []
    for c in labels:
        if c == len(unicodes):
            ret.append("")
        else:
            ret.append(unicodes[c])
    return "".join(ret)

def decode(yPred):  #Best Path, now beam search
    texts = []
    """
    for y in yPred:
        label = list(np.argmax(y[2:],1))
        label = [k for k, g in itertools.groupby(label)]
        text = labelsToText(label)
        texts.append(text)
    return texts
    """
    for i in range(yPred.shape[0]):
      y = yPred[i,2:,:]
      y = np.reshape(y,(1,30,97))
      pred =  K.get_value(K.ctc_decode(y, input_length=np.ones(y.shape[0])*30, greedy=False, beam_width=3, top_paths=1)[0][0])[0]
      word = ""
      for i in range(len(pred)):
        if pred[i] == len(unicodes):
          word += ""
        else:
          word += unicodes[pred[i]]
      texts.append(word)
    return texts

def preprocess(imag, imgSize=(128,32), dataAugmentation=0):
    imag = cv2.medianBlur(imag,3)
    J = cv2.resize(imag,(128,32), cv2.INTER_AREA)
    rt3,_ = cv2.threshold(J,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    N,_ = np.histogram(J,[i for i in range(0,256)])
    f = np.argmax(N[int(rt3):])
    rt4 = int(rt3+0*f) #use 0.7 instead of 0, if handwritten word images
    #for printed, 0 seemes to work the best
    _,k = cv2.threshold(J,rt4,255,cv2.THRESH_BINARY) 
    if dataAugmentation==1:
    	stretch = (random.random() - 0.5) # -0.5 .. +0.5
    	wStretched = max(int(k.shape[1] * (1 + stretch)), 1) # random width, but at least 1
    	k = cv2.resize(k, (wStretched, k.shape[0])) # stretch horizontally by factor 0.5 .. 1.5
    (wt, ht) = imgSize
    h = imag.shape[0]
    w = imag.shape[1]
    fx = w / wt #4
    fy = h / ht #3
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(k, newSize, cv2.INTER_AREA)#, cv2.INTER_AREA
    n,_ = np.histogram(img,[i for i in range(257)])
    nn = n[1:255]
    for j in range(len(nn)):
        if (np.sum(nn[:j])-np.sum(nn[j:]))/100 >= 0.25*np.sum(nn):
            break
    rt5 = j+1
    _,ii = cv2.threshold(img,rt5,255,cv2.THRESH_BINARY)
    H,W = ii.shape
    l,b = 32 - H, 128 - W
    if l%2 == 1:
        top, bottom = int(l/2) + 1, int(l/2)
    else:
        top, bottom = int(l/2), int(l/2)
    
    if b%2 == 1:
        left, right = int(b/2) + 1, int(b/2)
    else:
        left, right = int(b/2), int(b/2)
    FinalImage = cv2.copyMakeBorder(ii, top, bottom, left, right, cv2.BORDER_CONSTANT,255,255)
    #rotatedImg = cv2.rotate(FinalImage, cv2.ROTATE_90_CLOCKWISE) #change according to model
    rotatedImg = cv2.transpose(FinalImage)
    (m, s) = cv2.meanStdDev(rotatedImg)
    m = m[0][0]
    s = s[0][0]
    img = rotatedImg - m
    img = img / s if s>0 else img
    return np.reshape(img,(128,32))

#global graph
#graph = tf.compat.v1.get_default_graph()
model2 = getModel(False)
model2.load_weights('Models/Newtrainval_modelTranspose.hdf5') #change model here

def htr(filepath):
    image = cv2.imread(filepath,0) #change filename
    rows,cols = image.shape
    kernel = np.ones((9,9),np.uint8)
    erode = cv2.erode(image,kernel,iterations = 1)
    angle = determine_skew(erode)
    img = rotate(image, angle, resize=True) * 255
    img = np.uint8(img)
    print('\ngot image')
    
    # mser properties
    _delta=5
    _min_area=60
    _max_area=14400
    _max_variation=0.25
    _min_diversity=.2
    _max_evolution=200
    _area_threshold=1.01
    _min_margin=0.003
    _edge_blur_size=5
    
    mser = cv2.MSER_create(_delta,_min_area,_max_area,_max_variation,_min_diversity,_max_evolution,_area_threshold,_min_margin,_edge_blur_size)
    
    regions, boundingBoxes = mser.detectRegions(img)
    
    out_image_2 = np.zeros(img.shape,dtype='uint8')
    regions2 = []
    area_regions = []
    for region in regions:
        region = np.asarray(region)
        min1 = np.amin(region[:,0])
        max1 = np.amax(region[:,0])
        min2 = np.amin(region[:,1])
        max2 = np.amax(region[:,1])
        if max1 != min1 and max2 != min2:
            e = float(max2 - min2)/float(max1 - min1)
            ac = float(len(region))/((max2 - min2)*(max1 - min1))
            if e>0.1 and e<10 and ac>0.2:
                regions2.append(region)
                area_regions.append((max2 - min2)*(max1 - min1))
                out_image_2[ region[:,1] , region[:,0] ] = 255
    
    area_regions = np.asarray(area_regions)
    
    regions = regions2
    
    n, bins = np.histogram(area_regions,bins="auto")
    
    avg = 0
    num = 0
    
    a, b = bins[np.argmax(n)], bins[np.argmax(n)+1]
    for i in range(len(area_regions)):
        if area_regions[i]>a and area_regions[i]<b:
            avg += area_regions[i]
            num += 1
    avg = avg/float(num)
    
    kernell = np.ones((1,int(0.7*np.sqrt(avg))),np.uint8)
    appx_size = int(0.7*np.sqrt(avg))
    out_image_3 = cv2.dilate(out_image_2,kernell,iterations=1)
    kernel2 = np.ones((int(0.2*np.sqrt(avg)),1),np.uint8)
    out_image_4 = cv2.dilate(out_image_3,kernel2,iterations=1)
    
    cnts, _ = cv2.findContours(out_image_4.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    regions1 = []
    
    for i in range(len(cnts)):
        x,y,w,h = cv2.boundingRect(cnts[i])
        
        include = True
        
        for j in range(len(cnts)):
            if j!= i:
                x1,y1,w1,h1 = cv2.boundingRect(cnts[j])
                if x>=x1 and y>=y1 and x+w<=x1+w1 and y+h<=y1+h1:
                    include = False
    
        if (h>2*appx_size or w>2*appx_size or w*h>100) and include:
            regions1.append([x,y,w,h])
            
    regions1 = np.array(regions1)
    area = regions1[:,2] * regions1[:,3]
    area = np.array(sorted(area))/(rows*cols)
    
    regions2 = [[] for i in range(len(regions1))]
    regions2[0].append(regions1[0])
    line_idx = 0
    
    for i in range(1,len(regions1)):
        x,y,w,h = regions1[i]
        xa,ya,wa,ha = regions1[i-1]
        a = max(y,ya)
        b = min(h+y,ha+ya)
        if(b-a)>0:
            regions2[line_idx].append(regions1[i])
        else:
            line_idx = line_idx + 1
            regions2[line_idx].append(regions1[i]) 
    regions2 = np.array(regions2)
    regions2 = [x for x in regions2 if x != []]
    
    regions3 = []
    for i in range(len(regions2)-1,-1,-1):
        array = np.array(regions2[i])
        g = np.argsort(array[:,0])
        lin = array[g,:]
        regions3.append(lin)
    
    content = u''
    for line in regions3:
        LineString = ''
        for i in range(len(line[:,0])):
            x,y,w,h = line[i,:]
            w = img[y:y+h,x:x+w]
           # _, wordImage = cv2.threshold(word,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            Word = predict(w,model2)
            LineString += Word + '  '
        LineString += '\n'
        content += LineString
    
    return content


@app.route('/upload',methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"msg":"someting went wrong 1"})
        user_file = request.files['file']
        if user_file.filename == '':
            return "file name not found ..."
        print(user_file)
        path = os.path.join(os.getcwd(), user_file.filename)
        print(path)
        user_file.save(path)
        text = htr(path)
        print(text)
        os.remove(path)
        return text
        #return jsonify({'msg':text})
        
       

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=False,threaded=False) #,192.168.1.13
