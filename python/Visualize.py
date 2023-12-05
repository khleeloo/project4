import open3d as o3d



def visualize_point_cloud(mat_3D):
    '''takes in 3D np array and creates visualization'''

    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mat_3D)

    o3d.visualization.draw_geometries([pcd])