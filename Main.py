import pptk
import DataProcessing
from pathlib import Path
import numpy as np
import open3d as o3d

church_file = Path('Data/Church/Church.ply')
songo_mnara_file = Path('Data/SongoMnara/SongoMnara.ply')
songo_mnara_uds5_file = Path('Data/SongoMnara/SongoMnara_uds5.ply')
pointcloud = DataProcessing.load_from_ply(church_file)

pointcloud, segments = DataProcessing.segment_pointcloud(pointcloud, 5, segment_method='spatial', sort_axis='x')
xyz, intensity, rgb = DataProcessing.convert_to_arrays(pointcloud)

# Strip out center point
# augment points with approximated features
# Maybe multiscale features

# Could project down onto xy plane and use binary morphology to extract isolate clumps.

# pointcloud.uniform_down_sample(10)  # 10 seems max to preserve data, not random
# pointcloud.voxel_down_sample(1)  # Normals and colours are averaged (maybe don't want this behaviour...)
# pointcloud.voxel_down_sample_and_trace(1)
# more complex http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html#open3d.geometry.PointCloud.voxel_down_sample_and_trace


# Could save just the pointcloud and use a lambda function to convert to points
viewer = pptk.viewer(np.asarray(pointcloud.points))
viewer.set(lookat=(0, 0, 0))
viewer.attributes(np.asarray(pointcloud.colors), DataProcessing.flatten_list(segments))
# TODO: get open3D performant
