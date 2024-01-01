import open3d as o3d
import hdbscan
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

step_size = 0.01
dist_thr = 0.1
max_itr = 1000
ceilingThr = 0.30

def removeFloorCeilingPlanes(pointCloud):

    plane_model, inliers  = pointCloud.segment_plane(
        distance_threshold=dist_thr,
        ransac_n=3,
        num_iterations=max_itr
    )

    inlier_cloud = pointCloud.select_by_index(inliers)
    outlier_cloud = pointCloud.select_by_index(inliers, invert=True)

    plane_model_1, inliers_1  = outlier_cloud.segment_plane(
        distance_threshold=dist_thr,
        ransac_n=3,
        num_iterations=max_itr
    )

    inlier_cloud_1 = outlier_cloud.select_by_index(inliers_1)
    wall_points = outlier_cloud.select_by_index(inliers_1, invert=True)

    print(inlier_cloud, inlier_cloud_1)
    print(wall_points)

    # save point cloud data to pcd file
    o3d.io.write_point_cloud("Output/wall_points.pcd", wall_points)

    c1Max = np.asarray(inlier_cloud.get_max_bound())
    c2Max = np.asarray(inlier_cloud_1.get_max_bound())
    print(c1Max, c2Max)

    if c1Max[2] < c2Max[2]:
        o3d.io.write_point_cloud("Output/floor.pcd", inlier_cloud)
        o3d.io.write_point_cloud("Output/ceiling.pcd", inlier_cloud_1)
    else :
        o3d.io.write_point_cloud("Output/ceiling.pcd", inlier_cloud)
        o3d.io.write_point_cloud("Output/floor.pcd", inlier_cloud_1)
        
    return wall_points

def filter_point_cloud_z_range(point_cloud, z_min, z_max):
    """
    Filter a point cloud based on a specified range along the Z-coordinate.

    Parameters:
    - point_cloud: Open3D PointCloud object
    - z_min: Minimum value along the Z-coordinate
    - z_max: Maximum value along the Z-coordinate

    Returns:
    - filtered_point_cloud: Filtered Open3D PointCloud object
    """

    points = np.asarray(point_cloud.points)
    indices = np.where((points[:, 2] >= z_min) & (points[:, 2] <= z_max))[0]
    print(indices)
    
    filtered_point_cloud = point_cloud.select_by_index(indices)
    return filtered_point_cloud

def filter_point_cloud_x_range(point_cloud, x_min, x_max):
    points = np.asarray(point_cloud.points)
    indices = np.where((points[:, 0] >= x_min) & (points[:, 0] <= x_max))[0]
    
    filtered_point_cloud = point_cloud.select_by_index(indices)
    return filtered_point_cloud

def filter_point_cloud_y_range(point_cloud, y_min, y_max):

    points = np.asarray(point_cloud.points)
    indices = np.where((points[:, 1] >= y_min) & (points[:, 1] <= y_max))[0]
    
    filtered_point_cloud = point_cloud.select_by_index(indices)
    return filtered_point_cloud

def downSamplingVoxel(pointCloud, voxel_size):
    downPcd = pointCloud.voxel_down_sample(voxel_size)
    o3d.io.write_point_cloud("Output/down_sampling.pcd", downPcd)
    return downPcd

def removeOutliersFromWallPCD(pointCloud):
    ceiling = o3d.io.read_point_cloud("Output/ceiling.pcd")
    c1Min = np.asarray(ceiling.get_min_bound())
    print(c1Min)
    tmpAt30Ceiling = (c1Min[2]-0.05) - ceilingThr
    filtered = filter_point_cloud_z_range(pointCloud, tmpAt30Ceiling, c1Min[2] - 0.05 )
    o3d.io.write_point_cloud("Output/filtered.pcd", filtered)
    return filtered

def convert3DtoImg(pointCloud):
    grid = []
    max_points = 0
    c2Min = np.asarray(pointCloud.get_min_bound())
    c2Max = np.asarray(pointCloud.get_max_bound())
    print("c2Min: ", c2Min, " c2Max: ", c2Max)
    ax1Min, ax1Max, ax2Min, ax2Max = c2Min[0], c2Max[0], c2Min[1], c2Max[1]
    print(ax1Min, ax1Max, ax2Min, ax2Max)

    for i in np.arange(ax1Min, ax1Max, step_size):
        slicePoint = []
        sliceX = filter_point_cloud_x_range(pointCloud, i, i+ step_size)
        for j in np.arange(ax2Min, ax2Max, step_size):
            sliceY = filter_point_cloud_y_range(sliceX, j, j+ step_size)
            # print(sliceY)
            slicePoint.append(len(sliceY.points))
            if max_points <= len(sliceY.points) :
                max_points = len(sliceY.points)
            else :
                pass
        
        grid.append(slicePoint)

    print( "Show the image info: ", len(grid), max_points, len(grid[0]))    
    depthImg = np.zeros((len(grid), len(grid[0])),  dtype=np.uint8)
    
    for i in range(0, depthImg.shape[0]):
        for j in range(0, depthImg.shape[1]):
            percentOfMax = (grid[i][j] + 0.0) / (max_points + 0.0); 
            intensity = percentOfMax*255
            if 20 < intensity :
                depthImg[i][j] = 255;
            else :
                depthImg[i][j] = 0;
    return depthImg

def clusteringData(data):
    
    print(" *************** ")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    database_ = hdbscan.HDBSCAN(min_cluster_size = 100, min_samples= 5).fit(data_scaled)
    
    hlabels = database_.labels_
    hn_clusters_ = len(set(hlabels)) - (1 if -1 in hlabels else 0)
    hn_noise_ = list(hlabels).count(-1)
    print("Estimated number of clusters: %d" % hn_clusters_)
    print("Estimated number of noise points: %d" % hn_noise_)

    df_data = pd.DataFrame(data, columns=['x', 'y', 'z'])
    df_data.loc[:, "Cluster"] = hlabels

    clusterData = df_data.loc[ (df_data['Cluster'] != -1) ]
    clusterDataNp = clusterData[['x', 'y', 'z']].values
    print("ClusterDataNp ", len(clusterDataNp))
    return clusterDataNp

def statisticalRemovalOutlier(pointCloud, neighborNums, ratio):

    cl, ind = pointCloud.remove_statistical_outlier(
        nb_neighbors = neighborNums,
        std_ratio= ratio
    )
    print("After SOR filter : ", cl.points)
    
    return cl

def houghLineDetection(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 200, None, 5)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180 , 50,minLineLength = 100, maxLineGap = 10)
    
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(rgb_image,(x1,y1),(x2,y2),(0,255,0),2)
    
    cv2.imwrite('houghlines5.jpg', rgb_image)

def main():
    pcd = o3d.io.read_point_cloud("Input/Cropted_Point_index.pcd")
    wall_points =  removeFloorCeilingPlanes(pcd)
    wall_points = downSamplingVoxel(wall_points, 0.01)
    wall_points = removeOutliersFromWallPCD(wall_points)
    output = clusteringData(np.asarray(wall_points.points))

    # convert numpy to PointCloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(output)

    depthImg = convert3DtoImg(point_cloud)
    houghLineDetection(depthImg)


# pcd = o3d.io.read_point_cloud("Input/data.pcd")
# print(pcd)
# output = clusteringData(np.asarray(pcd.points))
    
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(output)
# o3d.io.write_point_cloud("Output/dataCluster.pcd", point_cloud)

if __name__=="__main__": 
    main()