# Structure from Motion


# 1. Using Superglue to support better feature matching
FeatureMatching.py
 Match the features
kp1,kp2,matches = ImageMatch(images[0], images[1])
print(len(matches))

# 2. Esential and Fubndamental Matrix
Below Code is found in main.py
```python
[F,mask]=cv2.findFundamentalMat(kp1,kp2, method=3,ransacReprojThreshold=3.0,confidence=0.99)
[E1,mask]=cv2.findEssentialMat(kp1,kp2,cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=3.0 )
[E2,mask]=cv2.findEssentialMat(kp2,kp1,cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=3.0 )
```


# 3. Pose Recovery
 ```python
[_, R1, t1, mask] = cv2.recoverPose(E1, kp1,kp2, cameraMatrix=K,mask=mask)    
[_, R2, t2, mask] = cv2.recoverPose(E2, kp2,kp1, cameraMatrix=K,mask=mask)  
projMatr1=np.column_stack((R1,t1))
projMatr2=np.column_stack((R2,t2))
```
R Recovered relative rotation, 3x3 matrix.
t Recovered relative translation, 3x1 vector.

mask Output mask for inliers in points1 and points2. In the output mask only inliers which pass the cheirality check. 
	
# 4. Triangulation

mat_4D=cv2.triangulatePoints(projMatr1=projMatr1,projMatr2=projMatr2,projPoints1=kp1,projPoints2=kp2).astype(np.float64)
```python
mat_3D = mat_4D[:,:3]/mat_4D[:,3:4]
```
changing to 3D according to since we require projection in 3D space.  they're just 3D points in a 4D projective space, analogous to 2D points in a 3D projective space. all points (x,y,z,1) * w, for arbitrary nonzero w, 
 in the projective space represent the same 3D point (x,y,z), and (x,y,z,1) is the canonical representative.
 https://stackoverflow.com/questions/69429075/what-could-be-the-reason-for-triangulation-3d-points-to-result-in-a-warped-para

#  5. PnP 
solvePnPRefineLM()

# 6. Optimization
Least squares with soft l1 loss: rho(z) = 2 * ((1 + z)**0.5 - 1)
```python
res1=least_squares(RMSE, r.flatten(),jac='2-point', method='dogbox',loss='soft_l1',max_nfev=2000) #implemented least square optimization
 
```

## 7. Visualization
Open3d point cloud
print("Load a ply point cloud, print it, and render it")
pcd = o3d.geometry.PointCloud()


## 8. Results


Datasets processed in the file Dataload.py

 ### TempleRing
![Temple Ring](results/templeRing1.png)
![Temple Ring](results/templeRing2.png)
![Temple Ring](results/templeRing3.png)
![Temple Ring](results/templeRing4.png)
 
 ### LLFF
 Intrinsic matrix constructed from pose information:

 ```python
  K=np.array([[fx,0,cx], [0 ,fy, cy],[0, 0, 1]]).astype(np.float64)  ##intrinsic matrix
```
# RMSE calculation

```python
def RMSE(r):
return np.power(r-r0.flatten(),2) ##error function for least squares
```
![TRex](results/trex/trex1.png)
![TRex](results/trex/trex2.png)
![TRex](results/trex/trex3.png)
![TRex](results/trex/trex4.png)
![TRex](results/trex/trex5.png)


![Fern](results/fern/fern1.png)
![Fern](results/fern/fern2.png)
![Fern](results/fern/fern3.png)
![Fern](results/fern/fern4.png)
