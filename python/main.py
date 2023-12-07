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

    
    i=1
    while  i<len(images):
        if poses is not None:
            pose_gt=poses[i] 
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
            print("Creating projection matrix for image {}.{}".format(i,z))
            # [_, R2, t2, mask] = cv2.recoverPose(E2, kp2,kp1, cameraMatrix=K,mask=mask)  
            r1,jacobian=cv2.Rodrigues(R1)


            CurrentProjMatr=np.column_stack((R1,t1))
            # projMatr2=np.column_stack((R2,t2))
            # CurrentProjMatr=np.dot(K,CurrentProjMatr)
            b=i-1
            print("Using previous projection matrix for image {}".format(b))
            PreviousProjMatr=proj_matr_list[i-1]
            # projMatr2=np.dot(K,projMatr2)
            mat_4D=triangulation(projMatr1=PreviousProjMatr,projMatr2=CurrentProjMatr,projPoints1=kp1,projPoints2=kp2).astype(np.float64)
            mat_3D = mat_4D[:3,:]


            points_2D=np.squeeze(kp1)
            points_3D=mat_3D.T

            dist_coeffs = np.zeros((4,1)).astype(np.float64)

            criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 20, 1e-6)
            r, t=PnP(points_3D,points_2D,K,dist_coeffs,r1,t1,criteria=criteria)
            R,jacobian=cv2.Rodrigues(r)
            pnp_mat=np.hstack([R,t])
            CurrentPose=pnp_mat

            mat_4D=triangulation(projMatr1=PreviousProjMatr,projMatr2=CurrentPose,projPoints1=kp1,projPoints2=kp2).astype(np.float64).T
            # #changing to 3D according to since we require projection in 3D space.  they're just 3D points in a 4D projective space, analogous to 2D points in a 3D projective space. all points (x,y,z,1) * w, for arbitrary nonzero w, 
            # # in the projective space represent the same 3D point (x,y,z), and (x,y,z,1) is the canonical representative.
            # # https://stackoverflow.com/questions/69429075/what-could-be-the-reason-for-triangulation-3d-points-to-result-in-a-warped-para
            mat_3D = mat_4D[:,:3]/mat_4D[:,3:4]
            # res1=least_squares(triangulation, CurrentPose.flatten(),args=(pose_gt.flatten(),kp1,kp2),jac='2-point', method='lm',loss='linear',max_nfev=2000)
            # print(res1)
            #visualize
            
            pcd.points = o3d.utility.Vector3dVector(mat_3D)
            mat_dict[n_inliners]=pcd
            # new_mat.append(pcd)

        proj_matr_list.append(CurrentPose)
        current_pose=CurrentPose.flatten()
        print(poses[i])
        pose_gt_str=poses[i].flatten()
        print(pose_gt_str)
        with open('results/kitti_trex_00.txt', 'a') as f: 
            for pose in current_pose:
                print(pose)
                f.write(str(pose)+' ')
            f.write('\n')
            f.close()
        with  open('results/kitty_trex_gt.txt', 'a') as w:
            for pose in pose_gt_str:
                w.write(str(pose)+' ')
            w.write('\n')
            w.close()
        i+=1

    # ##END LOOP  


    
    mat_dict=dict(sorted(mat_dict.items(),reverse=True))
    [new_mat.append(n) for _,n in mat_dict.items()]
    # o3d.visualization.draw_geometries(new_mat)
    o3d.visualization.draw_plotly(new_mat)

    return r,t, CurrentPose, pose_gt,  mat_dict

def train(path):

    new_mat=[]     
    i=1
    while i<100:
        r,t, CurrentPose, pose_gt, mat_dict=run_main_loop(path)

        def gt_r(pose_mat):
                # r_gt,_=cv2.Rodrigues(pose_mat[:, :3])
            return pose_mat.flatten()
        #least squares optimalization
        def fun(x):

            return np.subtract(np.abs(x),np.abs(gt_r(pose_gt)))
        print(CurrentPose.astype(np.int32), pose_gt.astype(np.int32))
        res1=least_squares(fun, CurrentPose.flatten(),jac='2-point', method='lm',loss='linear',max_nfev=2000)
        print(res1)
        i+=1
    mat_dict=dict(sorted(mat_dict.items(),reverse=True))
    [new_mat.append(n) for _,n in mat_dict.items()]
    # o3d.visualization.draw_geometries(new_mat)
    o3d.visualization.draw_plotly(new_mat)
    return res1['cost']

if __name__=="__main__":
    datadir='data/'
    img='trex/'
    # img='fern/'
    path=datadir+img
    # train(path)
    run_main_loop(path)
    

    

