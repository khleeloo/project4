import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import open3d as o3d
from FeatureMatching import ImageMatch

from Optimize import jac,least_squares

from Dataload import load_llff, load_templering
from Visualize import visualize_point_cloud
# from Optimize import jacobian_autograd
# import autograd.numpy as np
# from autograd import grad, jacobian
# from briefRotTest importdatadir





def run_main_loop(path):

    mat_dict={}
    proj_matr_list=[]
    ###Loading Datasets

    if 'templeRing' in path:
        images, K=load_templering(path)
    else:
        images,poses, K=load_llff(path)

    # print(K)
    # print(len(images))
    i=1
    #loading functions
    triangulation=cv2.triangulatePoints
    PnP=cv2.solvePnPRefineLM
    # PnP=cv2.solvePnP
    pcd = o3d.geometry.PointCloud()
    # Initialize
    print(len(images))
    kp1,kp2,matches,method, n_inliners= ImageMatch(images[1], images[2])
    if kp1.shape[0]>=3:
    
            # [F,mask]=cv2.findFundamentalMat(kp1,kp2, method=3,ransacReprojThreshold=3.0,confidence=0.99)
        [E1,mask]=cv2.findEssentialMat(kp1,kp2,cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=3.0 )
        [E2,mask]=cv2.findEssentialMat(kp2,kp1,cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=3.0 )
                #     # R Recovered relative rotation, 3x3 matrix.
                #     # t Recovered relative translation, 3x1 vector.
                #     # good the number of inliers which pass the cheirality check.
                #     # mask Output mask for inliers in points1 and points2. In the output mask only inliers which pass the cheirality check. Vector of length N, see the Mask input option.
                #     # triangulatedPoints 3D points which were reconstructed by triangulation, see cv.triangulatePoints


        [_, R1, t1, mask] = cv2.recoverPose(E1, kp1,kp2, cameraMatrix=K,mask=mask)    
        [_, R2, t2, mask] = cv2.recoverPose(E2, kp2,kp1, cameraMatrix=K,mask=mask)  
        r1,jacobian=cv2.Rodrigues(R1)
        projMatr1=np.column_stack((R1,t1))
        # projMatr2=np.column_stack((R2,t2))
        projMatr1=np.dot(K,projMatr1)
        proj_matr_list.append(projMatr1)
        # projMatr02=np.dot(K,projMatr2)



        # mat_4D=triangulation(projMatr1=projMatr01,projMatr2=projMatr02,projPoints1=kp1,projPoints2=kp2).astype(np.float64)
        # mat_3D = mat_4D[:3,:]
        # pcd.points = o3d.utility.Vector3dVector(mat_3D.T)
        # mat_dict[n_inliners]=pcd

        new_mat=[]
        projMatr1=None
    else:
        pass
    
    while  i<len(images[:2]):

        for z in range(0,len(images)-1):    
            k1,k2,matches,method, n_inliners= ImageMatch(images[i],images[z])
        
            if method is None:
                z+=1
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
            print("Creating projection matrix for image {}.{}".format(i,z))
            # [_, R2, t2, mask] = cv2.recoverPose(E2, kp2,kp1, cameraMatrix=K,mask=mask)  
            r1,jacobian=cv2.Rodrigues(R1)

            # if projMatr1 is None:
           
            # else:
            CurrentProjMatr=np.column_stack((R1,t1))
            # projMatr2=np.column_stack((R2,t2))
            CurrentProjMatr=np.dot(K,CurrentProjMatr)
            b=i-1
            print("Using previous projection matrix for image {}".format(b))
            PreviousProjMatr=proj_matr_list[i-1]
            # projMatr2=np.dot(K,projMatr2)
            mat_4D=triangulation(projMatr1=PreviousProjMatr,projMatr2=CurrentProjMatr,projPoints1=kp1,projPoints2=kp2).astype(np.float64)
            mat_3D = mat_4D[:3,:]
        
            # # R Recovered relative rotation, 3x3 matrix.
        #     # t Recovered relative translation, 3x1 vector.
        #     # good the number of inliers which pass the cheirality check.
        #     # mask Output mask for inliers in points1 and points2. In the output mask only inliers which pass the cheirality check. Vector of length N, see the Mask input option.
        #     # triangulatedPoints 3D points which were reconstructed by triangulation, see cv.triangulatePoints

                # # Given the correspondence and the projective matrix, the 3D point can be computed by
        # # triangulation via DLT algorithm.
        # # You can use the function in opencv and Matlab to recover the camera pose for triangu-
        # # lation.
            

        # # Given 2D-3D correspondences and the intrinsic parameter, estimate a camera pose using
        # # linear least squares. You can find the material on the PnP in the course slides.
        # # You can use the function in OpenCV and Matlab to recover the camera pose for trian-
        # # gulation.
            # points_2D=np.squeeze(kp1, axis=1)
            points_2D=np.squeeze(kp1)
            n=points_2D.shape[0]
            points_3D=mat_3D.T
            # col=np.zeros((n,1)).astype(np.float64)
            # points_3D=np.column_stack((points_2D,col))
            dist_coeffs = np.zeros((4,1)).astype(np.float64)

            criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 20, 1e-6)
            r, t=PnP(points_3D,points_2D,K,dist_coeffs,r1,t1,criteria=criteria)
            R,jacobian=cv2.Rodrigues(r)
            pnp_mat=np.hstack([R,t])
            CurrentMatr=np.dot(K,pnp_mat)
            

            mat_4D=triangulation(projMatr1=PreviousProjMatr,projMatr2=CurrentMatr,projPoints1=kp1,projPoints2=kp2).astype(np.float64).T
            # #changing to 3D according to since we require projection in 3D space.  they're just 3D points in a 4D projective space, analogous to 2D points in a 3D projective space. all points (x,y,z,1) * w, for arbitrary nonzero w, 
            # # in the projective space represent the same 3D point (x,y,z), and (x,y,z,1) is the canonical representative.
            # # https://stackoverflow.com/questions/69429075/what-could-be-the-reason-for-triangulation-3d-points-to-result-in-a-warped-para
            mat_3D = mat_4D[:,:3]/mat_4D[:,3:4]
        
            #visualize
            
            pcd.points = o3d.utility.Vector3dVector(mat_3D)
            mat_dict[n_inliners]=pcd
            # new_mat.append(pcd)

        proj_matr_list.append(CurrentMatr)
        i+=1
    ##END LOOP  

    # print(new_mat)
    mat_dict=dict(sorted(mat_dict.items(),reverse=True))
    [new_mat.append(n) for m ,n in mat_dict.items()]
    o3d.visualization.draw_geometries(new_mat)
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(new_mat)
    # vis.update_geometry()
    # vis.poll_events()
    # vis.update_renderer()
    # vis.capture_screen_image('results/img.png')
    # vis.destroy_window()

        


if __name__=="__main__":
    datadir='data/'
    # img='fern/'
    img='templeRing/'
    path=datadir+img
    run_main_loop(path)

    

