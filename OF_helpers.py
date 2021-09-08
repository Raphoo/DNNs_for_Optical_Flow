import numpy as np
print('numpy version: %s'%np.__version__)
import matplotlib # for plotting
import matplotlib.pyplot as plt

import sys
print('python version: %s'%sys.version)
import cv2 # opencv

def visualize_flow(img):
    hsv = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.uint8)
    flow_magnitude, flow_angle = cv2.cartToPolar(img[0,:,:], img[1,:,:])
    hsv[:,:, 0] = flow_angle * 180 / np.pi / 2
    hsv[:,:, 2] = 255
    hsv[:,:, 1] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    visualize_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    #print(visualize_img.shape)
    #so confused. so this is 3 dimensions for color, 
    
    return visualize_img

def downsample_image(img): #should probably use ndarray? or can it cut tensors..?
    """
    img = [batchsize, channo, L, L]
    returns img,img1,img2,img3 with L continually halved.
    """
    batchsize,channo,L,L1 = img.shape
    img1 = [[cv2.resize(img[j,i,:,:],(L//2,L1//2)) for i in range(channo)] for j in range(batchsize)]
    img2 = [[cv2.resize(img[j,i,:,:],(L//4,L1//4)) for i in range(channo)] for j in range(batchsize)]
    img3 = [[cv2.resize(img[j,i,:,:],(L//8,L1//8)) for i in range(channo)] for j in range(batchsize)]
    
    img1 = np.array(img1)
    img2 = np.array(img2)
    img3 = np.array(img3)
    return img,img1,img2,img3
    

def test_images(X,Y1,i,Y=None):
    print("Sanity Check! Does OF line up with input?")

    print(Y1.shape)
    print(X.shape)
 

    plt.figure()
    plt.subplot(221)
    plt.imshow(X[i,0,:,:]) #yay! loaded in Z.

    plt.subplot(222)
    plt.imshow(X[i,1,:,:])

    plt.subplot(223, title='after tanh')
    plt.imshow(visualize_flow(Y1[i,:,:,:]))
    
    if Y is not None:
        plt.subplot(224,title='before tanh')
        plt.imshow(visualize_flow(Y[i,:,:,:]))

    plt.tight_layout()
    plt.show()

    print("This is the", i,"th image.")


def shuffle_OF(X,Y,Y1=None): #here, Y1 would be the tanh'd version of Y.
    #or you can input Y as the tanh'd version, and not shuffle the original.
    #up to you!
    shuffler = np.random.permutation(X.shape[0])
    X = X[shuffler,:,:,:]
    Y = Y[shuffler,:,:,:]
    if Y1: Y1 = Y1[shuffler,:,:,:]

    if not Y1: return X,Y
    else: return X,Y,Y1


def visualize_midway(inp,pred,ground_truth,ep,bt):
    """
    in the middle of training; execute to visualize how we're doing!
    """

    plt.figure()
    plt.subplot(221)
    plt.imshow(inp[0,0,:,:]) #random resized images!

    plt.subplot(222)
    plt.imshow(inp[0,1,:,:])

    plt.subplot(223,title=('LABEL Epoch: ' + str(ep) + ' Batch: ' +str(bt)))
    plt.imshow(visualize_flow(ground_truth[0,:,:,:])) #true OF corresponding to img
    
    plt.subplot(224,title=('PRED Epoch: ' + str(ep) + ' Batch: ' +str(bt)))
    plt.imshow(visualize_flow(pred[0,:,:,:])) #predicted OF corresponding to img


    plt.tight_layout()
    plt.show()