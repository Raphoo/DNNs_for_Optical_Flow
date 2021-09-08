import random
import numpy as np
import numbers
import types
import scipy.ndimage as ndimage


def random_rotate(inputs,target,angle,diff_angle=0,order=2,reshape=False):#input is [batchsize,2,Lxd,Lyd] and lbl is [batchsize,2,Lxd,Lyd]

    """Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    interpolation order: Default: 2 (bilinear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0.
    """
    applied_angle = random.uniform(-angle,angle)
    diff = random.uniform(-diff_angle,diff_angle)
    angle1 = applied_angle - diff/2
    angle2 = applied_angle + diff/2
    angle1_rad = angle1*np.pi/180
    diff_rad = diff*np.pi/180

    h, w, _ = target.shape 
    #so basically, if I just extract the correct width and height then
    #I can modify this code to cater to my 4d input target shape.
    #my target shape is [batchsize, 2, w, h.]

    warped_coords = np.mgrid[:w, :h].T + target
    warped_coords -= np.array([w / 2, h / 2])

    warped_coords_rot = np.zeros_like(target)

    warped_coords_rot[..., 0] = \
        (np.cos(diff_rad) - 1) * warped_coords[..., 0] + np.sin(diff_rad) * warped_coords[..., 1]

    warped_coords_rot[..., 1] = \
        -np.sin(diff_rad) * warped_coords[..., 0] + (np.cos(diff_rad) - 1) * warped_coords[..., 1]

    target += warped_coords_rot
    
    new_inputs = []

    new_inputs.append(ndimage.interpolation.rotate(inputs[0], angle1)) #yeah so i guess this truly is weird,
    #how would numpy array fit?
    new_inputs.append(ndimage.interpolation.rotate(inputs[1], angle2))
    target = ndimage.interpolation.rotate(target, angle1)
    # flow vectors must be rotated too! careful about Y flow which is upside down
    target_ = np.copy(target)
    target[:,:,0] = np.cos(angle1_rad)*target_[:,:,0] + np.sin(angle1_rad)*target_[:,:,1]
    target[:,:,1] = -np.sin(angle1_rad)*target_[:,:,0] + np.cos(angle1_rad)*target_[:,:,1]
    
    new_inputs = np.array(new_inputs)
    return new_inputs,target



def center_crop(size,inputs,target):
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size
    """
    h1, w1 = inputs[0].shape
    h2,w2 = h1,w1
    th, tw = size
    x1 = int(round((w1 - tw) / 2.))
    y1 = int(round((h1 - th) / 2.))
    x2 = int(round((w2 - tw) / 2.))
    y2 = int(round((h2 - th) / 2.))
    
    new_inputs = []
    new_inputs.append(inputs[0][y1: y1 + th, x1: x1 + tw])
    new_inputs.append(inputs[1][y2: y2 + th, x2: x2 + tw])
    new_inputs = np.array(new_inputs)
    target = target[y1: y1 + th, x1: x1 + tw]
    return new_inputs,target


def do_rotated_center_crop(img,lbl,angle=0,size=144):
    #img dims: [bathsize,2,:,:]
    batchsize = img.shape[0]
    new_img = np.zeros((img.shape[0],2,size,size))
    new_lbl = np.zeros((img.shape[0],2,size,size))
    for p in range(batchsize):
        Y_batch = np.moveaxis(lbl[p,:,:,:],0,2)
        X_batch = img[p,:,:,:]

        #print(X_batch.shape,Y_batch.shape)
        X2,Y2 = random_rotate(X_batch,Y_batch,angle)
        X3,Y3 = center_crop((size,size),X2,Y2)

        Y3 = np.moveaxis(Y3,2,0)
        
        
        new_img[p] = X3
        new_lbl[p] = Y3
    return new_img, new_lbl

def do_rotated_random_crop(img,lbl,angle=0,size=144):
    batchsize =  img.shape[0]
    
    
    #img dims: [bathsize,2,:,:]
    
    new_img = np.zeros((img.shape[0],2,size,size))
    new_lbl = np.zeros((img.shape[0],2,size,size))
    for p in range(batchsize):
        Y_batch = np.moveaxis(lbl[p,:,:,:],0,2)
        X_batch = img[p,:,:,:]

        #print(X_batch.shape,Y_batch.shape)
        X2,Y2 = random_rotate(X_batch,Y_batch,angle)
        Y2 = np.moveaxis(Y2,2,0)

        _,h,w = X2.shape

        cuth = np.random.randint(0,h-size-1)
        cutw = np.random.randint(0,w-size-1)  

        X3 = X2[:,cuth:cuth+size,cutw:cutw+size]
        Y3 = Y2[:,cuth:cuth+size,cutw:cutw+size]
        
        
        new_img[p] = X3
        new_lbl[p] = Y3
    return new_img, new_lbl
    

def check_import(number):
    print(number, "import worked!")