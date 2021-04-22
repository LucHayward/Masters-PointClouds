import pptk
import DataProcessing
from pathlib import Path
import numpy as np
import open3d as o3d


# xyz, intensity, rgb, pointcloud = DataProcessing.main()
# church_file = Path('/Users/luc/Downloads/MastersDatasets/Church/Church.ply')
church_file = Path('Data/Church/Church.ply')
xyz, intensity, rgb, pointcloud = DataProcessing.load_from_ply(church_file)

pointcloud, segments = DataProcessing.segment_pointcloud(pointcloud, 5)





# pointcloud.uniform_down_sample(10)  # 10 seems max to preserve data, not random
# pointcloud.voxel_down_sample(1)  # Normals and colours are averaged (maybe don't want this behaviour...)
# pointcloud.voxel_down_sample_and_trace(1)  # more complex http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html#open3d.geometry.PointCloud.voxel_down_sample_and_trace


# Could save just the pointcloud and use a lambda function to convert to points
viewer = pptk.viewer(xyz)
viewer.attributes(rgb, segments)
# TODO: get open3D performant
