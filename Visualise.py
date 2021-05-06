from pathlib import Path

import pptk

import DataProcessing

import numpy as np
import open3d as o3d


def display_inlier_outlier_o3d(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def compare_pointcloud_after_cleaning(pointcloud):
    pcd, ind = DataProcessing.clean_pointcloud(pointcloud, True)
    xyz, pcd_xyz = np.asarray(pointcloud.points), np.asarray(pcd.points)
    rgb = np.asarray(pointcloud.colors)
    pcd_i = 0
    removed = []
    for i in range(len(xyz)):
        if len(pcd_xyz) > pcd_i and (xyz[i] == np.asarray(pcd.points)[pcd_i]).all():
            removed.append(0)
            pcd_i = pcd_i + 1
        else:
            removed.append(1)
    print(f"Started with {len(xyz)} points, after cleaning {len(pcd_xyz)} points remain.\n"
          f"Removed {len(xyz)-len(pcd_xyz)} = {(len(xyz)-len(pcd_xyz))/len(xyz)}% of points.")
    incorrect_removal = [removed[i] == 1 and rgb[i, 0] != 1 for i in range(len(removed))]
    print(f"Incorrectly removed {incorrect_removal.count(True)} = {incorrect_removal.count(True)/len(xyz)}% of points ({incorrect_removal.count(True)/(len(xyz)-len(pcd_xyz))} % of removed points).")

    # Change the colours to .25 alhpa gray for the points that were not incorrectly removed and highlight the rest blue
    incorrect_removal = [(0.5,0.5,0.5,0.05) if not i else (1,0,0,1) for i in incorrect_removal]
    v = pptk.viewer(xyz, rgb, removed, incorrect_removal)


if __name__ == '__main__':
    church_file = Path('Data/Church/Church.ply')
    xyz, intensity, rgb, pointcloud = DataProcessing.load_from_ply(church_file)

    pointcloud, segments = DataProcessing.segment_pointcloud(pointcloud, 5)
    xyz, intensity, rgb = DataProcessing.convert_to_arrays(pointcloud)

    if input("Visualise Church dataset using uniform point segmentation? (y/n)").lower() == 'y':
        viewer = pptk.viewer(xyz)
        viewer.attributes(rgb, segments)
    if input("Visualise Church Dataset using spatially uniform segmentation (y/n)").lower() == 'y':
        axis = input("Axis x/y/z")
        assert axis in ['x', 'y', 'z']
        pointcloud, segments = DataProcessing.segment_pointcloud(pointcloud, 5, 'spatial', sort_axis=axis)
        xyz, intensity, rgb = DataProcessing.convert_to_arrays(pointcloud)

        viewer = pptk.viewer(xyz)
        viewer.attributes(rgb, DataProcessing.flatten_list(segments))
