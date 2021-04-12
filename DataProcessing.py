import numpy as np
import pptk
import open3d as o3d
from pathlib import Path


def load_from_ply(file):
    """
    Loads the ply file in to memory
    :param file: PathLib file to .ply file
    :return: xyz, intensity, rgb and open3d point cloud
    """
    print(f"Loading {file}")
    filename = str(ply_files[index])
    ply_load = o3d.io.read_point_cloud(filename)
    xyz = np.asarray(ply_load.points)
    rgb = np.asarray(ply_load.colors)
    intensity = np.asarray(ply_load.normals)[:, 0]
    return xyz, intensity, rgb, ply_load


def load_from_ptx(file_list):
    """
    Loads data from the give ptx files into memory
    :param file_list: PathLib ptx files
    :return: xyz, intensity and rgb
    """
    # Read in the data from text
    point_clouds = []
    rows = []
    cols = []
    for ptx_file in [x for x in file_list if x.suffix == ".ptx"]:
        with open(ptx_file, 'r') as file:
            print(f"File: {ptx_file.name}")
            row, col = [int(next(file).strip()) for _ in range(2)]
            rows.append(row)
            cols.append(col)
            print(f'Rows: {row}\nCols: {col}\nTotal points: {row * col}\n')
            point_clouds.append(np.loadtxt(ptx_file, skiprows=10))
    Flatten = lambda t: [item for sublist in t for item in sublist]
    flat_point_cloud = Flatten(point_clouds)
    xyz = flat_point_cloud[:, :3]
    intensity = flat_point_cloud[:, 3]
    rgb = flat_point_cloud[:, 4:]
    return xyz, intensity, rgb


def save_to_ply(file, pcd):
    """
    Given the parameters for the point cloud, saves them to the given file in PCD format
    :param file: Pathlib file to store to
    :param pcd: open3d Point Cloud
    :return: if the file write succeeds
    """
    return o3d.io.write_point_cloud(str(file), pcd, print_progress=True)


def convert_to_pointcloud( xyz, intensity, rgb):
    """
    Creates an open3d Point Cloud from the given parameters and saves the intensity in the first normal channel
    :param xyz:
    :param intensity:
    :param rgb:
    :return: the open3d point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255)
    # storing intensity in normals[0]
    pcd.normals = o3d.utility.Vector3dVector(
        np.stack((intensity, np.zeros(intensity.shape), np.zeros(intensity.shape)), 1))
    return pcd


datasets_dir = Path("/Users/luc/Downloads/MastersDatasets/")
sub_dirs = [x for x in datasets_dir.iterdir() if x.is_dir()]
print(f"Available datasets:\n{sub_dirs}")
dataset_dir = sub_dirs[int(input("Directory index: "))]
files_list = [x for x in dataset_dir.iterdir()]
files_list.sort()
print(files_list)
ply_files = [x for x in files_list if x.suffix == '.ply']

if any(ply_files) and input(f"Found .ply files:\n{ply_files}\nLoad (y/n)?: "):
    index = int(input("File index:"))
    xyz, intensity, rgb, pointcloud = load_from_ply(ply_files[index])

else:
    xyz, intensity, rgb = load_from_ptx(files_list)
    pointcloud = convert_to_pointcloud(xyz, intensity, rgb)
    save_to_ply(dataset_dir / (dataset_dir.name + ".ply"), xyz, intensity, rgb)

# TODO multithread
# point_cloud = np.loadtxt(file_to_open, skiprows=10)
print("Loaded point clouds")

classification = [x == 255 for x in rgb[:, 0]]


# save_to_ply(dataset_dir / (dataset_dir.name + ".ply"), xyz, intensity, rgb)
