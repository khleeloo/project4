import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

from FeatureMatching import ImageMatch

from Optimize import jac,least_squares

from Dataload import load_llff, load_templering
from Visualize import visualize_point_cloud
# from Optimize import jacobian_autograd
# import autograd.numpy as np
# from autograd import grad, jacobian
# from briefRotTest importdatadir


# Match the features
        # kp1,kp2,matches = ImageMatch(images[0], images[1])
        # print(len(matches))
        # [F,mask]=cv2.findFundamentalMat(kp1,kp2, method=3,ransacReprojThreshold=3.0,confidence=0.99)
        # [E1,mask]=cv2.findEssentialMat(kp1,kp2,cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=3.0 )
        # [E2,mask]=cv2.findEssentialMat(kp2,kp1,cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=3.0 )
        #     # R Recovered relative rotation, 3x3 matrix.
        #     # t Recovered relative translation, 3x1 vector.
        #     # good the number of inliers which pass the cheirality check.
        #     # mask Output mask for inliers in points1 and points2. In the output mask only inliers which pass the cheirality check. Vector of length N, see the Mask input option.
        #     # triangulatedPoints 3D points which were reconstructed by triangulation, see cv.triangulatePoints


        # [_, R1, t1, mask] = cv2.recoverPose(E1, kp1,kp2, cameraMatrix=K,mask=mask)    
        # [_, R2, t2, mask] = cv2.recoverPose(E2, kp2,kp1, cameraMatrix=K,mask=mask)  
        # projMatr1=np.column_stack((R1,t1))
        # projMatr2=np.column_stack((R2,t2))

        # # Given the correspondence and the projective matrix, the 3D point can be computed by
        # # triangulation via DLT algorithm.
        # # You can use the function in opencv and Matlab to recover the camera pose for triangu-
        # # lation.
            
        # triangulation=cv2.triangulatePoints()
        # PnP=cv2.solvePnP()
        # mat_4D=triangulation(projMatr1=projMatr1,projMatr2=projMatr2,projPoints1=kp1,projPoints2=kp2).astype(np.float64)
        # #changing to 3D according to since we require projection in 3D space.  they're just 3D points in a 4D projective space, analogous to 2D points in a 3D projective space. all points (x,y,z,1) * w, for arbitrary nonzero w, 
        # # in the projective space represent the same 3D point (x,y,z), and (x,y,z,1) is the canonical representative.
        # # https://stackoverflow.com/questions/69429075/what-could-be-the-reason-for-triangulation-3d-points-to-result-in-a-warped-para
        # mat_3D = mat_4D[:3,:]




        # # Given 2D-3D correspondences and the intrinsic parameter, estimate a camera pose using
        # # linear least squares. You can find the material on the PnP in the course slides.
        # # You can use the function in OpenCV and Matlab to recover the camera pose for trian-
        # # gulation.
        # points_2D=kp1
        # mat_3D=mat_3D.T
        # dist_coeffs = np.zeros((4,1)).astype(np.float64)
        # success, vector_rotation, vector_translation=PnP(mat_3D,points_2D,K,dist_coeffs,R1,t1)


        # visualize_point_cloud(mat_3D)

def run_main_loop(path):


    ###Loading Datasets

    if 'templeRing' in path:
        images, K=load_templering(path)
    else:
        imgages=load_llff(path)
    i=len(images)-1

    #loading functions
    triangulation=cv2.triangulatePoints
    PnP=cv2.solvePnP
    new_mat=None

    while  i>0:

        if i>=1:
            #feature matching
            
            kp1,kp2,matches,method= ImageMatch(images[i-1], images[i])
            print(method)
            if method is None:
                i-=1
                continue
            elif method == '7_point':
                [F,mask]=cv2.findFundamentalMat(kp1,kp2, method=cv2.FM_7POINT,ransacReprojThreshold=3.0,confidence=0.99)
                
                
                [E1,mask]=cv2.findEssentialMat(kp1,kp2,cameraMatrix=K, method=cv2.FM_7POINT, prob=0.999, threshold=3.0 )
       
                [E2,mask]=cv2.findEssentialMat(kp2,kp1,cameraMatrix=K, method=cv2.FM_7POINT, prob=0.999, threshold=3.0 )
            else:
                #fundamental matrix
                [F,mask]=cv2.findFundamentalMat(kp1,kp2, method=3,ransacReprojThreshold=3.0,confidence=0.99)
                
                
                [E1,mask]=cv2.findEssentialMat(kp1,kp2,cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=3.0 )
                # cv2.findEssentialMat
                [E2,mask]=cv2.findEssentialMat(kp2,kp1,cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=3.0 )
        
            [_, R1, t1, mask] = cv2.recoverPose(E1, kp1,kp2, cameraMatrix=K,mask=mask)    

            [_, R2, t2, mask] = cv2.recoverPose(E2, kp2,kp1, cameraMatrix=K,mask=mask)  
            projMatr1=np.column_stack((R1,t1))
            projMatr2=np.column_stack((R2,t2))
            # # R Recovered relative rotation, 3x3 matrix.
        #     # t Recovered relative translation, 3x1 vector.
        #     # good the number of inliers which pass the cheirality check.
        #     # mask Output mask for inliers in points1 and points2. In the output mask only inliers which pass the cheirality check. Vector of length N, see the Mask input option.
        #     # triangulatedPoints 3D points which were reconstructed by triangulation, see cv.triangulatePoints

                # # Given the correspondence and the projective matrix, the 3D point can be computed by
        # # triangulation via DLT algorithm.
        # # You can use the function in opencv and Matlab to recover the camera pose for triangu-
        # # lation.
            
            mat_4D=triangulation(projMatr1=projMatr1,projMatr2=projMatr1,projPoints1=kp1,projPoints2=kp2).astype(np.float64)
            # #changing to 3D according to since we require projection in 3D space.  they're just 3D points in a 4D projective space, analogous to 2D points in a 3D projective space. all points (x,y,z,1) * w, for arbitrary nonzero w, 
            # # in the projective space represent the same 3D point (x,y,z), and (x,y,z,1) is the canonical representative.
            # # https://stackoverflow.com/questions/69429075/what-could-be-the-reason-for-triangulation-3d-points-to-result-in-a-warped-para
            mat_3D = mat_4D[:3,:]
      
        # # Given 2D-3D correspondences and the intrinsic parameter, estimate a camera pose using
        # # linear least squares. You can find the material on the PnP in the course slides.
        # # You can use the function in OpenCV and Matlab to recover the camera pose for trian-
        # # gulation.
            points_2D=kp1
            mat_3D=mat_3D.T
            dist_coeffs = np.zeros((4,1)).astype(np.float64)
            success, vector_rotation, vector_translation=PnP(mat_3D,points_2D,K,dist_coeffs,R1,t1)


        else:
            pass

        if new_mat is None:
            new_mat=mat_3D
        else:
            new_mat=np.hstack([new_mat,mat_3D])
            
        i-=1

    visualize_point_cloud(new_mat.T)
    ##END LOOP  





            

       




    

        





if __name__=="__main__":
    datadir='principles/project4/data/'
    img='templeRing/'
    path=datadir+img
    run_main_loop(path)

    

