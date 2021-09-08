from PIL import Image

import numpy as np
from numpy import asarray
import cv2
from scipy import io
import matplotlib.pyplot as plt


import os

import importlib
import optic_flow_utils
importlib.reload(optic_flow_utils)
from optic_flow_utils import flow_read, flow_to_img

def get_pairs(Z): #will be running subsets through this.
    """
    Put into list of pairs for optic flow algorithms.
    """
    Z_pairs = [[i,j] for i,j in zip(Z[0:-1], Z[1:])]
    
    
    #print("Length of image pairs {}".format(len(Z_pairs)))
    
    return Z_pairs


def get_partial_X(Z): #here, Z is just partial. For all imgs in folder.

    Z_pairs = get_pairs(Z) 

    X = np.zeros((len(Z_pairs), 2, Lyd, Lxd*2))
    X[:,0,:,:] = np.stack([i[0].reshape(Lyd, Lxd*2) for i in Z_pairs])
    X[:,1,:,:] = np.stack([i[1].reshape(Lyd, Lxd*2) for i in Z_pairs])
    
    return X #so in this case, X should be 2*Z -1 in length.


def cut_half(img): #so I know this will come in the form of X_temp or Y_temp.
                    #that is to say, input is 4D array: #of things, 2 chans, Lyd, Lxd.
        
    height, width = img[0,0].shape
    width_cutoff = width // 2

    left = img[:,:,:, :width_cutoff]
    right = img[:,:,:,width_cutoff:]

        
    half_x = np.concatenate([left,right],axis=0)
    
        
    return half_x


def load_sintel_data(path,Lyd=218,Lxd=256):
    """
    Takes in path to sintel dataset.
    Loads in all images, cuts them in half, returns X(input images) and Y(labels)
    reshaped to specified dimensions.
    X and Y have dimensions (# of data, 2, Lyd, Lxd).

    """

    names = [filename for filename in os.listdir(path)]
    print("folder names in directory:",names)


    number_of_flo = sum([len(files) for r, d, files in os.walk(path)])
    print(number_of_flo)

    number_of_img = number_of_flo + len(names)
    print(number_of_img)

    X = np.zeros((number_of_flo*2, 2, Lyd, Lxd)) #*2 because i will cut imgs in half!
    Y = np.zeros((number_of_flo*2, 2, Lyd, Lxd))

    #Lyd, and Lxd are my desired downsampled dimensions.

    #I am going to fill X with paired off Z's,
    #and Y with ground truth flow truncated in half!
    #will resize/rotate/etc. inside of the training loop later :)


    bigcount = 0

    for j in range(len(names)): #iterate through all training folders
        
        #pair off Z inside of this loop, and extract flo at same time, make sure end of one
        #folder does not bleed into next.
        
        path_img = 'MPI-Sintel-complete/training/clean/' + names[j] + '/'
        path_flo = 'MPI-Sintel-complete/training/flow/' + names[j] + '/'
        
        num_imgs_inpath = len([name for name in os.listdir(path_img)])
        
        Z_temp = np.zeros((num_imgs_inpath, Lyd,Lxd*2))
        #Z_temp will be filled with full size imgs, not half.
        #will truncate at X step.
                        
        Y_temp = np.zeros((num_imgs_inpath-1, 2, Lyd,Lxd*2))
        
        
        #making a temporary Z, to pair off and fill X!
        
        for i in range(1,num_imgs_inpath+1):
            
            #for 50 images in folder, y should go from 1->49.
            
            
            img = Image.open(path_img + 'frame_{:04}.png'.format(i))
            img = img.convert("L") #just making it b&w.

            img = np.asarray(img)
            
            
            img = cv2.resize(img,(Lxd*2,Lyd))
            
            
            Z_temp[i-1] = img
            
            if i<num_imgs_inpath: #because there is one less flow file!
                
                flow = flow_read(path_flo + 'frame_{:04}.flo'.format(i))
                flow = np.moveaxis(flow,2,0) # moving channel axis to front.
                
                newflow = np.zeros((2,Lyd,Lxd*2))
                
                newflow[0] = cv2.resize(flow[0,:,:],(Lxd*2,Lyd))
                newflow[1] = cv2.resize(flow[1,:,:],(Lxd*2,Lyd))
                
                
                Y_temp[i-1] = newflow 
                
                
            
        X_temp = get_partial_X(Z_temp) #X_temp is 4D array.
        X_temp = cut_half(X_temp)
        Y_temp = cut_half(Y_temp)
        
        print(X_temp.shape,range(bigcount,bigcount+num_imgs_inpath*2-2))
        
        
        #here, I can technically resize the images...!
        #method = cv2.resize(image,(newxdim,newydim))
        

        X[bigcount:bigcount+num_imgs_inpath*2-2] = X_temp
        Y[bigcount:bigcount+num_imgs_inpath*2-2] = Y_temp

        bigcount = bigcount+num_imgs_inpath*2-2
        #all the images paired off, then truncated = 2(num-1) total.

                
                
    print(bigcount) #count should be same as 2*number_of_files; just making sure.
    return X,Y 



def load_chairs(path_chair,start,end,Lyd=216,Lxd=256): #start = k, end=k+batch
    """
    Load in chairs images!
    I have made it take in a "start" and "end" argument so we could batch
    load in images during training, as my kernel was being too crashy otherwise.
    If you just want the X and Y of the first 100 images, you can input 0 and 100, etc.
    """
    num_chair_files = 22872
    X_c1 = np.zeros((end-start, 2, Lyd, Lxd)) #*2 because i will cut imgs in half!
    Y_c1 = np.zeros((end-start, 2, Lyd, Lxd))

    for j in range(start+1,end+1):

        #well, chair dimensions: 384x512.

        img1 = Image.open(path_chair + '{:05}_img1.ppm'.format(j))
        img1 = img1.convert("L") #just making it b&w.
        img1 = asarray(img1)
        img1 = cv2.resize(img1,(Lxd,Lyd))

        img2 = Image.open(path_chair + '{:05}_img2.ppm'.format(j))
        img2 = img2.convert("L") #just making it b&w.
        img2 = asarray(img2)
        img2 = cv2.resize(img2,(Lxd,Lyd))
        
        X_c1[j-1,0,:,:] = img1
        X_c1[j-1,1,:,:] = img2



        flow = flow_read(path_chair + '{:05}_flow.flo'.format(j))
        flow = np.moveaxis(flow,2,0) # moving channel axis to front.

        newflow = np.zeros((2,Lyd,Lxd))

        newflow[0] = cv2.resize(flow[0,:,:],(Lxd,Lyd))
        newflow[1] = cv2.resize(flow[1,:,:],(Lxd,Lyd))

        Y_c1[j-1] = newflow
        
    #Y1_c1 = np.tanh(Y_c1/30)  #if you want to make flow more vivid!
        
    return X_c1,Y1_c1