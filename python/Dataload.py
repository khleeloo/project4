import numpy as np
from load_llff import load_llff_data
import os
import cv2

def load_llff(path):
    ####LLFF 

    images, poses, bds, render_poses, i_test = load_llff_data(path, factor=8, recenter=True, bd_factor=.75)
    hwf = poses[0,:3,-1]

    ##hwf is the parameter for "height width focal"

    ## fx = fy = f in this case.
    fx=hwf[2]
    #  Cx = W/2 and Cy = H /2 
    cx=hwf[1]/2
    cy=hwf[0]/2

    poses = poses[:,:3,:4]





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

    