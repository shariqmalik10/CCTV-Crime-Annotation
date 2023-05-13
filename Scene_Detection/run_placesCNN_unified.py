# PlacesCNN to predict the scene category, attribute, and class activation map in a single pass
# by Bolei Zhou, sep 2, 2017
# updated, making it compatible to pytorch 1.x in a hacky way
# Transfer learning and resolution performed by Shariq Nadeem

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import numpy as np
import cv2
from PIL import Image
# import argparse
# import time
# import tensorflow_hub as hub
# import math
# import re
# from basicsr.archs.rrdbnet_arch import RRDBNet
# from scenedetect import detect, ContentDetector

# from cv2 import dnn_superres
from Scene_Detection.image_super_resolution.ISR.models import RDN, RRDN
import Scene_Detection.wideresnet as wideresnet

#realesgran inference
# def realesgran(upload):
    # os.system("!python3 inference_realesrgan.py -n RealESRGAN_x4plus -i" + upload + "--outscale 3.5") #type: ignore

working_dir = "C:/Users/Tee/Desktop/FYP/GitFYP/Crime_Annotation"

#slice one frame from video 
def vid_slice(vid_filename):
    video = cv2.VideoCapture(vid_filename)
    i = 0
    #get the number of frames per second or fps
    fps = video.get(cv2.CAP_PROP_FPS)
    #get number of frames
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    #get the total duration of video in seconds
    seconds = round(frames / fps)
    #for now we set it to take frame from the first minute of every video 
    frame_id = frame_id = int(fps*(1*(seconds//2)+0))
    #extract the frame
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = video.read()
    # store the frame in a temp jpg file
    frame_filename = "original_image.jpg"
    cv2.imwrite(frame_filename, frame)

    # return frame_filename

    #experimenting
    return frame

    
#clearing the resolution
# hacky way to deal with the Pytorch 1.0 update
def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1 # type: ignore
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

def load_labels():
    # prepare all the labels
    # scene category relevant
    file_name_category = working_dir + "/Scene_Detection/categories_places365.txt"
    # if not os.access(file_name_category, os.W_OK):
    #     synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    #     os.system('wget ' + synset_url)
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = working_dir + '/Scene_Detection/IO_places365.txt'
    # if not os.access(file_name_IO, os.W_OK):
    #     synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
    #     os.system('wget ' + synset_url)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = working_dir + '/Scene_Detection/labels_sunattribute.txt'
    # if not os.access(file_name_attribute, os.W_OK):
    #     synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
    #     os.system('wget ' + synset_url)
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = working_dir + '/Scene_Detection/W_sceneattribute_wideresnet18.npy'
    # if not os.access(file_name_W, os.W_OK):
    #     synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
    #     os.system('wget ' + synset_url)
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute

def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def returnTF():
# load the image transformer
    tf = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf


def load_model():
    # this model has a last conv feature map as 14x14

    model_file = working_dir + "/Scene_Detection/wideresnet18_places365.pth.tar"
    # if not os.access(model_file, os.W_OK):
    #     os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
    #     os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')

    
    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    
    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
    

    model.eval()
    # hook the feature extractor
    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature) #type: ignore
    return model


# load the model
features_blobs = []
model = load_model()

def scene_predict(filename, model_name="none"):
    '''
    model_name : This is used to specify if a super resolution model is being used 
    current options are : psnr-small, psnr-large, rrdn
    '''
    # load the labels
    classes, labels_IO, labels_attribute, W_attribute = load_labels()

    # load the transformer
    tf = returnTF() # image transformer

    # get the softmax weight
    params = list(model.parameters()) # type: ignore
    weight_softmax = params[-2].data.numpy()
    weight_softmax[weight_softmax<0] = 0

    frame_file = vid_slice(filename)
    #isr

    #preprocess image. convert to numpy 
    # img = Image.open(frame_file) # type: ignore

    #experimenting
    img = frame_file

    # Selecting the super resolution model
    #convert numpy to array regardless of network used or even if there is no network 
    if model_name != "none" and model_name != "realesgran":
        lr_img = np.array(img)
        if model_name=="psnr-small":
            supmodel = RDN(weights='psnr-small')
        elif model_name == "rrdn":
            supmodel = RRDN(weights='gans')
        elif model_name == "noise-cancel":
            supmodel = RDN(weights='noise-cancel')
        elif model_name == "psnr-large":
            supmodel = RDN(weights='psnr-large')
        #may raise unbound error however can be ignored as it will enter the statement only if user selects a model for super resolution 
        sr_img = supmodel.predict(lr_img) #type: ignore
        img = Image.fromarray(sr_img)
    elif model_name=="none":
        img = Image.fromarray(img)

    #normalize input 
    input_img = V(tf(img)).unsqueeze(0)
    cv2.imwrite("output_image.jpg", np.asarray(img)) 
    # forward pass
    logit = model.forward(input_img) # type: ignore
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # output the IO prediction
    io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor

    #determine whether indoor or outdoor environment using io_image
    environment = ""
    if io_image < 0.5:
        environment = "indoor"
    else:
        environment = "outdoor"

    # output the prediction of scene category
    print('--SCENE CATEGORIES:')
    scene_catgeories = []
    categories = []
    for i in range(0, 7):
        #given the experiments, there is strong evidence to suggest that the model is heavily biased towards the classes
        #home_theatre, movie_theatre. So we skip over those predictions for this particular model 
        if classes[idx[i]] != "home_theater" and classes[idx[i]] != "movie_theater/indoor":
            scene_catgeories.append(('{:.3f} -> {}'.format(probs[i], classes[idx[i]])))
            categories.append(classes[idx[i]])
        # print('{:.3f}x -> {}'.format(probs[i], classes[idx[i]]))

    # print(scene_catgeories)
    # print(categories)
    # output the scene attributes
    responses_attribute = W_attribute.dot(features_blobs[1])
    idx_a = np.argsort(responses_attribute)
    print('--SCENE ATTRIBUTES:')
    print(', '.join([labels_attribute[idx_a[i]] for i in range(-1,-10,-1)]))
    attributes = [labels_attribute[idx_a[i]] for i in range(-1,-10,-1)]
    print(attributes)
    
    #refine the catgeory output to have only text and spaces.
    # category_prediction = categories[0]
    # for char in category_prediction:
    #     if (char.isalpha() != True):
    #         category_prediction = category_prediction.replace(char, " ")

    top5catgeory = categories[0:6]
    append_catgeory =[]
    for cat in top5catgeory:
        for char in cat:
            if (char.isalpha() != True):
                cat = cat.replace(char, " ")
                append_catgeory.append(cat)
    

    
    #make a result dictionary and store important attributes that need to be output in the UI
    result_dict = {
        "environment": environment,
        "scene_category": append_catgeory,
        "attribute_1": attributes[0],
        "attribute_2": attributes[2],
        "attribute_3": attributes[3]
    }

    #print result dictionary 
    print("Prediction with model: " + model_name)
    print(result_dict)

    return result_dict


# print(scene_predict('/Users/shariqmalik/Documents/3rdYS2/FYP/Scene_Detection_Algorithm/Anomaly-Videos-Part-1/Abuse/Abuse002_x264.mp4', "rrdn"))
