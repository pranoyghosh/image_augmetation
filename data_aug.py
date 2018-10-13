import cv2
import numpy as np
import random
import time
start_time = time.time()

#functions for augmentation
def random_rotation(image):
    (r,c)= image.shape[:2]
    M = cv2.getRotationMatrix2D((c/2,r/2),180,1.0)
    dst1 = cv2.warpAffine(image,M,(c,r))
    return dst1

def random_noise(image):
    row,col,ch= image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy

def horizontal_flip(image):
    return image[:, ::-1]

def scale(image):
    res = cv2.resize(image,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    return res

def crop(image):
    img1= image[image.shape[0]/4:3*image.shape[0]/4,image.shape[1]/4:3*image.shape[1]/4]
    return img1

def translate(image):
    rows,cols,_ = image.shape
    M = np.float32([[1,0,150],[0,1,50]])
    dst = cv2.warpAffine(image,M,(cols,rows))
    return dst

def vertical_flip(image):
    rimg=cv2.flip(image,1)
    return rimg

def shear1(image):
    rows,cols,ch = image.shape
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(image,M,(cols,rows))
    return dst

def shear2(image):
    rows,cols,ch = image.shape
    pts2 = np.float32([[50,50],[200,50],[50,200]])
    pts1 = np.float32([[10,100],[200,50],[100,250]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(image,M,(cols,rows))
    return dst

folder_path = 'User/Desktop/img_aug/Dataset'
num_files_desired = 50
num_generated_files = 0


# dictionary of the transformations we defined earlier
available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip,
    'scaling': scale,
    'cropping': crop,
    'translating' : translate,
    'vertical_flip' : vertical_flip,
    'shear1' : shear1,
    'shear2' : shear2
}

img=cv2.imread('image.jpg',1)


while num_generated_files <= num_files_desired:
    transformed_image = None
    key = random.choice(list(available_transformations))
    transformed_image = available_transformations[key](img)
    new_file_path = '%s/augmented_image_%s.jpg' % (folder_path, num_generated_files)
    cv2.imwrite(new_file_path, transformed_image)
    num_generated_files += 1

print("--- %s seconds ---" % (time.time() - start_time))
