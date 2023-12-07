import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import open3d as o3d
from FeatureMatching import ImageMatch
from scipy.optimize import leastsq, curve_fit, least_squares

# from Optimize import jac,least_squares

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
    poses=None

    if 'templeRing' in path:
        images, K=load_templering(path)
    else:
        images,poses, bds,K=load_llff(path)



    #loading functions
    triangulation=cv2.triangulatePoints
    PnP=cv2.solvePnPRefineLM
    # PnP=cv2.solvePnP
    pcd = o3d.geometry.PointCloud()

    
    # Initialize

    kp1,kp2,matches,method, n_inliners= ImageMatch(images[0], images[1])

    
        # [F,mask]=cv2.findFundamentalMat(kp1,kp2, method=3,ransacReprojThreshold=3.0,confidence=0.99)
    [E1,mask]=cv2.findEssentialMat(kp1,kp2,cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=3.0 )
    [E2,mask]=cv2.findEssentialMat(kp2,kp1,cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=3.0 )

    [_, R1, t1, mask] = cv2.recoverPose(E1, kp1,kp2, cameraMatrix=K,mask=mask)                    #     # R Recovered relative rotation, 3x3 matrix.
    r1,jacobian=cv2.Rodrigues(R1)
    projMatr1=np.column_stack((R1,t1))
    projMatr1=np.dot(K,projMatr1)
    proj_matr_list.append(projMatr1)

    new_mat=[]
    projMatr1=None

    RMSE_r=0
    RMSE_t=0
    i=1
    loss=0
    while  i<len(images):
        if poses is not None:
            pose_gt=poses[i] 
            r0,_=cv2.Rodrigues(pose_gt[:, :3]) # computing rotation vec  ground truth for llff
            t0=pose_gt[:, 3] # computing translation vec ground truth for llff
            bds_gt=bds[i]

        for z in range(0,len(images)-1):    
            kp1,kp2,matches,method, n_inliners= ImageMatch(images[i],images[z])
        
            if method is None:
                z+=1
                continue
            elif method == '7_point':
                # [F,mask]=cv2.findFundamentalMat(kp1,kp2, method=cv2.FM_7POINT,ransacReprojThreshold=3.0,confidence=0.99)
                
                
                [E1,mask]=cv2.findEssentialMat(kp1,kp2,cameraMatrix=K, method=cv2.FM_7POINT, prob=0.999, threshold=3.0 )
        
                [E2,mask]=cv2.findEssentialMat(kp2,kp1,cameraMatrix=K, method=cv2.FM_7POINT, prob=0.999, threshold=3.0 )
            else:
                #fundamental matrix
                # [F,mask]=cv2.findFundamentalMat(kp1,kp2, method=3,ransacReprojThreshold=3.0,confidence=0.99)
                
                
                [E1,mask]=cv2.findEssentialMat(kp1,kp2,cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=3.0 )

                [E2,mask]=cv2.findEssentialMat(kp2,kp1,cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=3.0 )
        
            [_, R1, t1, mask] = cv2.recoverPose(E1, kp1,kp2, cameraMatrix=K,mask=mask)    
            print("Creating projection matrix for image {}.".format(i))
            # [_, R2, t2, mask] = cv2.recoverPose(E2, kp2,kp1, cameraMatrix=K,mask=mask)  
            r1,_=cv2.Rodrigues(R1)

            CurrentProjMatr=np.column_stack((R1,t1))

            CurrentProjMatr=np.dot(K,CurrentProjMatr)
            b=i-1
            print("Using previous projection matrix for image {}".format(b))
            PreviousProjMatr=proj_matr_list[i-1]

            mat_4D=triangulation(projMatr1=PreviousProjMatr,projMatr2=CurrentProjMatr,projPoints1=kp1,projPoints2=kp2).astype(np.float64)
            mat_3D = mat_4D[:3,:]


            points_2D=np.squeeze(kp1)
            points_3D=mat_3D.T

            dist_coeffs = np.zeros((4,1)).astype(np.float64)

            criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 20, 1e-6)
            r, t=PnP(points_3D,points_2D,K,dist_coeffs,r1,t1,criteria=criteria)

            RMSE_r+=np.power(sum(r.flatten())-sum(r0.flatten()),2) #calculate RMSE r and update
            RMSE_t+=np.power(sum(t.flatten())-sum(t0.flatten()),2)  #calculate RMSE t and update

            def RMSE(r):
                return np.power(r-r0.flatten(),2) ##error function for least squares
            
            res1=least_squares(RMSE, r.flatten(),jac='2-point', method='dogbox',loss='soft_l1',max_nfev=2000) #implemented least square optimization
            loss+=res1['cost']
            r=res1['x']

            R,jacobian=cv2.Rodrigues(r)
            pnp_mat=np.hstack([R,t])
            CurrentPose=np.dot(K,pnp_mat)

            mat_4D=triangulation(projMatr1=PreviousProjMatr,projMatr2=CurrentPose,projPoints1=kp1,projPoints2=kp2).astype(np.float64).T
            # #changing to 3D according to since we require projection in 3D space.  they're just 3D points in a 4D projective space, analogous to 2D points in a 3D projective space. all points (x,y,z,1) * w, for arbitrary nonzero w, 
            # # in the projective space represent the same 3D point (x,y,z), and (x,y,z,1) is the canonical representative.
            # # https://stackoverflow.com/questions/69429075/what-could-be-the-reason-for-triangulation-3d-points-to-result-in-a-warped-para
            mat_3D = mat_4D[:,:3]/mat_4D[:,3:4]

        #visualize
        
        pcd.points = o3d.utility.Vector3dVector(mat_3D)
        mat_dict[n_inliners]=pcd
        # new_mat.append(pcd)

        proj_matr_list.append(CurrentPose)
  
        current_pose_str=CurrentPose.flatten()
        pose_gt_str=poses[i].flatten()

        with open('results/kitti_trex_00.txt', 'a') as f:   ###saving results for odometry evaluation
            k=''
            for p in current_pose_str:
                print(p)
               
                k+=str(p)+' '
            f.write(k.strip())
            f.write('\n')
            f.close()
        with  open('results/kitti_trex_gt.txt', 'a') as w:  ###saving results for odometry evaluation
            z=''
            for pose in pose_gt_str:
                z+=str(pose)+' '
            w.write(z.strip())
            w.write('\n')
            w.close()
        i+=1
        
    TOTAL_RMSE_r =  np.sqrt( (1/len(images)**2)* RMSE_r)
    TOTAL_RMSE_t =  np.sqrt( (1/len(images)**2)* RMSE_t)
    mean_loss=loss/len(images)**2
    print('TOTAL_RMSE_r',TOTAL_RMSE_r)    
    print('TOTAL_RMSE_t',TOTAL_RMSE_t)   
    print('Average loss',mean_loss)   
    # ##END LOOP  


    
    mat_dict=dict(sorted(mat_dict.items(),reverse=True))
    [new_mat.append(n) for _,n in mat_dict.items()]
    # o3d.visualization.draw_geometries(new_mat)
    o3d.visualization.draw_plotly(new_mat)

    return r,t, CurrentPose, pose_gt,  mat_dict


if __name__=="__main__":
    datadir='data/'
    # img='trex/'
    img='fern/fern/'
    path=datadir+img
 
    run_main_loop(path)
    

    

