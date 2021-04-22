import numpy as np
import open3d as o3d
from pathlib import Path


def load_from_ply(file):
    """
    Loads the ply file in to memory
    :param file: PathLib file to .ply file
    :return: xyz, intensity, rgb and open3d point cloud
    """
    print(f"Loading {file}")
    filename = str(file)
    ply_load = o3d.io.read_point_cloud(filename)
    xyz = np.asarray(ply_load.points)
    rgb = np.asarray(ply_load.colors)
    intensity = np.asarray(ply_load.normals)[:, 0]
    return xyz, intensity, rgb, ply_load


def load_from_ptx(file_list):
    """
    Loads data from the give ptx files into memory. Converts red channel to 0/255
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

    flat_point_cloud = np.concatenate(point_clouds)
    # xyz = flat_point_cloud[:, :3]
    # intensity = flat_point_cloud[:, 3]
    # rgb = flat_point_cloud[:, 4:] / 255
    # flat_point_cloud = flat_point_cloud[:, 4] * 255
    flat_point_cloud[:, 4] = [255 if x != 0 else 0 for x in flat_point_cloud[:, 4]]
    return flat_point_cloud[:, :3], flat_point_cloud[:, 3], flat_point_cloud[:, 4:] / 255


def save_to_ply(file, pcd):
    """
    Given the parameters for the point cloud, saves them to the given file in PCD format
    :param file: Pathlib file to store to
    :param pcd: open3d Point Cloud
    :return: if the file write succeeds
    """
    return o3d.io.write_point_cloud(str(file), pcd, print_progress=True)


def get_spatial_info(pointcloud):
    print(f"Max bounds: {pointcloud.get_max_bound()}")
    print(f"Min bounds: {pointcloud.get_min_bound()}")
    print(f"Mean and covariance: {pointcloud.compute_mean_and_covariance()}")
    print(f"Nearest Neighbour distance: {pointcloud.computer_nearest_neighbour_distance()}")


def clean_pointcloud(pointcloud):
    # pointcloud.remove_statistical_outlier(neighbours, ratio)
    # pointcloud.remove_radius_outlier(n_points, radius)
    raise NotImplementedError


# TODO: allow shape
# def segment_pointcloud(pointcloud, shape):
def segment_pointcloud(pointcloud, num_splits):
    # if len(shape) > 3:
    #     print("Cannot use greater than 3 dimensions")
    print(f"Splitting pointcloud in {num_splits}")
    xyz, rgb, intensity = convert_to_arrays(pointcloud)
    x_sorted_inds = xyz[:, 0].argsort()  # get the indices for xyz sorted on x
    xyz, rgb, intensity = xyz[x_sorted_inds], rgb[x_sorted_inds], intensity[x_sorted_inds]


    # TODO segments are very eneven spatially (first and last cover much more ground)
    class_segments = np.array_split(rgb[:,0], num_splits)
    xyz_segments = np.array_split(xyz, num_splits)
    print(f"Split {len(rgb)} points along the x axis into {num_splits} chunks of size:\n"
          f"{[len(s) for s in class_segments]}\n"
          f"Num 'removed' points per chunk:\n"
          f"{[np.sum(s) for s in class_segments]}\n"
          f"x-distance in each chunk:\n"
          f"{[seg[:,0].max() - seg[:,0].min() for seg in xyz_segments]}\n"
          f"Area of each chunk:\n"
          f"{[(seg[:,1].max() - seg[:,1].min()) * (seg[:,0].max() - seg[:,0].min()) for seg in xyz_segments]}")




    # for i, s in enumerate(segments):
    #     for ss in s:
    #         ss[0] = i

    return convert_to_pointcloud(xyz, intensity, rgb), class_segments


def generate_segment_mask(size, splits):
    return [x//size for x in range(size)]


def convert_to_arrays(pointcloud):
    return np.asarray(pointcloud.points), np.asarray(pointcloud.colors), np.asarray(pointcloud.normals)[:, 0]


def convert_to_pointcloud(xyz, intensity, rgb):
    """
    Creates an open3d Point Cloud from the given parameters and saves the intensity in the first normal channel
    :param xyz:
    :param intensity:
    :param rgb:
    :return: the open3d point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    # storing intensity in normals[0]
    pcd.normals = o3d.utility.Vector3dVector(
        np.stack((intensity, np.zeros(intensity.shape), np.zeros(intensity.shape)), 1))
    return pcd


def main():
    """
    Opens the MastersDataset folder and loads the chosen datasets into memory
    :return: xyz, intensity and RGB data as well as the o3d PointCloud object
    """
    datasets_dir = Path("Data/")
    sub_dirs = [x for x in datasets_dir.iterdir() if x.is_dir()]
    print(f"Available datasets:\n{sub_dirs}")
    dataset_dir = sub_dirs[int(input("Directory index: "))]
    files_list = [x for x in dataset_dir.iterdir()]
    files_list.sort()
    print(files_list)
    ply_files = [x for x in files_list if x.suffix == '.ply']

    if any(ply_files) and input(f"Found .ply files:\n{ply_files}\nLoad (y/n)?: ") == "y":
        index = int(input("File index:")) if len(ply_files) > 1 else 0
        xyz, intensity, rgb, pointcloud = load_from_ply(ply_files[index])

    else:
        xyz, intensity, rgb = load_from_ptx(files_list)
        pointcloud = convert_to_pointcloud(xyz, intensity, rgb)
        save_to_ply(dataset_dir / (dataset_dir.name + ".ply"), pointcloud)

    print("Loaded point clouds")

    classification = [x == 255 for x in rgb[:, 0]]
    return xyz, intensity, rgb, pointcloud
    # save_to_ply(dataset_dir / (dataset_dir.name + ".ply"), xyz, intensity, rgb)


if __name__ == '__main__':
    main()
