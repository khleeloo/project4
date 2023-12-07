import numpy as np
from load_llff import load_llff_data
import os
import cv2

def load_image(path):
    images = []
    datadir=os.path.dirname(path + 'images_4/')
    print(datadir)
    for filename in os.listdir(datadir):
        if filename.endswith('.png'):
            img = cv2.imread(datadir+'/'+filename)
                # Turn into grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            if img is not None:
                images.append(img)
    return images

def load_llff(path):
    ####LLFF 

    images, poses, bds, render_poses, i_test = load_llff_data(path, factor=8, recenter=True, bd_factor=.75)
    hwf = poses[0,:3,-1]

    ##hwf is the parameter for "height width focal"

    ## fx = fy = f in this case.
    fx=hwf[2]
    fy=fx
    #  Cx = W/2 and Cy = H /2 
    cx=hwf[1]/2
    cy=hwf[0]/2

    K=np.array([[fx,0,cx], [0 ,fy, cy],[0, 0, 1]]).astype(np.float64)
    poses = poses[:,:3,:4]
    # llff_list=[]
    # for img in images:
    #     img = np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(np.float64)
    #     if img is not None:
    #         llff_list.append(img)
    llff_list=load_image(path)


    return llff_list, poses,bds, K





# ##Temple Ring
# # Load the images
def load_templering(path):
    images = []
    datadir=os.path.dirname(path)

    K = np.genfromtxt(datadir + '/camera.txt', dtype=str).astype(np.float64)
    for filename in os.listdir(datadir):
        if filename.endswith('.png'):
            img = cv2.imread(path+filename)
                # Turn into grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            if img is not None:
                images.append(img)
    return images, K

    