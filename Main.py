import pptk
import DataProcessing
from pathlib import Path

# xyz, intensity, rgb, pointcloud = DataProcessing.main()
church_file = Path('/Users/luc/Downloads/MastersDatasets/Church/Church.ply')
xyz, intensity, rgb, pointcloud = DataProcessing.load_from_ply(church_file)
viewer = pptk.viewer(xyz)
viewer.attributes(rgb)
# TODO: get open3D performant

