import numpy as np
import open3d as o3d
from pathlib import Path

from scipy import stats

axis_dict = {'x': 0, 'y': 1, 'z': 2}


def load_from_ply(file):
    """
    Loads the ply file in to memory
    :param file: PathLib file to .ply file
    :return: xyz, intensity, rgb and open3d point cloud
    """
    print(f"Loading {file}")
    filename = str(file)
    ply_load = o3d.io.read_point_cloud(filename)
    # xyz = np.asarray(ply_load.points)
    # rgb = np.asarray(ply_load.colors)
    # intensity = np.asarray(ply_load.normals)[:, 0]
    return ply_load


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
            pcd = np.loadtxt(ptx_file, skiprows=10)
            pcd = np.delete(pcd, np.where(np.all(pcd[:, :3] == 0, axis=1)), axis=0)
            point_clouds.append(pcd)

    flat_point_cloud = np.concatenate(point_clouds)

    # xyz = flat_point_cloud[:, :3]
    # intensity = flat_point_cloud[:, 3]
    # rgb = flat_point_cloud[:, 4:] / 255
    # flat_point_cloud = flat_point_cloud[:, 4] * 255
    flat_point_cloud[flat_point_cloud[:, 4] == 1, 4] = 255
    # flat_point_cloud[:, 4] = [255 if x != 0 else 0 for x in flat_point_cloud[:, 4]]
    return flat_point_cloud[:, :3], flat_point_cloud[:, 3], flat_point_cloud[:, 4:] / 255


def save_to_ply(file, pcd):
    """
    Given the parameters for the point cloud, saves them to the given file in PCD format
    :param file: Pathlib file to store to
    :param pcd: open3d Point Cloud
    :return: if the file write succeeds
    """
    return o3d.io.write_point_cloud(str(file), pcd, print_progress=True)


def print_spatial_info(pointcloud):
    print(f"Max bounds: {pointcloud.get_max_bound()}")
    print(f"Min bounds: {pointcloud.get_min_bound()}")
    print(f"Mean and covariance: {pointcloud.compute_mean_and_covariance()}")
    print(f"Nearest Neighbour distance: {pointcloud.computer_nearest_neighbour_distance()}")


def print_stat_info(pointcloud):
    xyz, intensity, rgb = convert_to_arrays(pointcloud)
    print("XYZ stats")
    stdev, mean, median = np.std(xyz, axis=0), np.mean(xyz, axis=0), np.median(xyz, axis=0)
    mode = stats.mode(xyz, axis=0)
    print(f"Standard deviation: {stdev}")
    print(f"Mean: {mean}")
    print(f"Median: {median}")


def clean_pointcloud(pointcloud, visualise=False, num_neighbours=20, std_ratio=20):
    """
    Remove outliers and unnecessary points
    TODO NotYetImplemented
    :param num_neighbours:
    :param std_ratio:
    :param visualise:
    :param pointcloud:
    :return: cleaned pointcloud
    """
    # pointcloud.remove_statistical_outlier(neighbours, ratio)
    # pointcloud.remove_radius_outlier(n_points, radius)
    xyz, intensity, rgb = convert_to_arrays(pointcloud)
    # TODO remove severely outlying points

    zscore = stats.zscore(xyz, axis=0)  # Calculate stdevs from mean
    print(f"Zscore(xyz):\n{zscore}")
    mean_point = np.mean(xyz, axis=0)
    distances = np.linalg.norm(np.asarray(pointcloud.points) - mean_point, axis=0)

    print(f"Removing statistical outliers with open3d(nb_neighbours={num_neighbours}, std_ratio{std_ratio})")
    # Very CPU bound
    pcd, ind = pointcloud.remove_statistical_outlier(nb_neighbors=num_neighbours, std_ratio=std_ratio)
    print("Visualising removed points")

    import Visualise
    if visualise: Visualise.display_inlier_outlier_o3d(pointcloud, ind)
    # TODO remove high duplication points
    return pcd, ind


def find_nearest_id(array, value):
    import math
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def remove_xyz_points(pointcloud, xyz):
    inds = [i for i in range(len(np.asarray(pointcloud.points))) if (np.asarray(pointcloud.points)[i] == xyz).all()]
    return pointcloud.select_by_index(inds, invert=True)


def remove_zero_points(pointcloud):
    """
    Removes the (0,0,0) points from the pointcloud used as standins for unfound points
    :param pointcloud:
    :return: a pointcloud with the zero points removed
    """
    return remove_xyz_points(pointcloud, (0, 0, 0))


# TODO: allow shape
# def segment_pointcloud(pointcloud, shape):
def segment_pointcloud(pointcloud, num_splits, segment_method='uniform', sort_axis='x'):
    """

    :param pointcloud:
    :param num_splits:
    :param segment_method: ['uniform', 'spatial', 'grid']
    :param sort_axis:
    :return: Pointcloud, list of segment_ids
    """
    assert segment_method in ['uniform', 'spatial', 'grid']
    # if len(shape) > 3:
    #     print("Cannot use greater than 3 dimensions")
    # TODO refactor sorting
    print(f"Splitting pointcloud in {num_splits if segment_method != 'grid' else num_splits * num_splits}")
    xyz, intensity, rgb = sort_pointcloud(pointcloud, sort_axis)

    # TODO segments are very uneven spatially (first and last cover much more ground)
    label_segments, xyz_segments = [], []
    if segment_method == 'uniform':
        label_segments = np.array_split(rgb[:, 0].copy(), num_splits)
        xyz_segments = np.array_split(xyz.copy(), num_splits)
    elif segment_method == 'spatial':
        total_distances = pointcloud.get_max_bound() - pointcloud.get_min_bound()
        intervals = total_distances // num_splits

        interval_vals = [xyz[0, axis_dict[sort_axis]] + intervals[axis_dict[sort_axis]] * i for i in
                         range(1, num_splits + 1)]
        interval_idxs = [find_nearest_id(xyz[:, axis_dict[sort_axis]], v) for v in interval_vals]

        # TODO handle splits < 3?
        label_segments = [rgb[:interval_idxs[0], 0].copy()] + \
                         [rgb[interval_idxs[i]:interval_idxs[i + 1], 0].copy() for i in range(len(interval_idxs) - 2)]
        label_segments.append(rgb[interval_idxs[-2]:, 0].copy())
        xyz_segments = [xyz[:interval_idxs[0]]].copy() + \
                       [xyz[interval_idxs[i]:interval_idxs[i + 1]].copy() for i in range(len(interval_idxs) - 2)]
        xyz_segments.append(xyz[interval_idxs[-2]:].copy())
    elif segment_method == 'grid':
        xyz_max, xyz_min = pointcloud.get_max_bound(), pointcloud.get_min_bound()
        total_distances = xyz_max - xyz_min
        intervals = total_distances // num_splits
        xs = np.arange(pointcloud.get_min_bound()[0], pointcloud.get_max_bound()[0], intervals[0])
        ys = np.arange(pointcloud.get_min_bound()[1], pointcloud.get_max_bound()[1], intervals[1])
        grid_shape = (len(xs) - 1, len(ys) - 1)
        grid = np.zeros(grid_shape).tolist()  # Contains all the point_idxs for that grid cell
        total = 0
        for x in range(num_splits):
            for y in range(num_splits):
                print(f'cell ({x},{y})')
                if x == num_splits - 1:
                    grid[x][y] = np.where(
                        (xyz[:, 0] >= (xyz_min[0] + intervals[0] * x)) &
                        # (xyz[:, 0] <= (xyz_min[0] + intervals[0] * (x + 1))) &
                        (xyz[:, 1] >= (xyz_min[1] + intervals[1] * y)) &
                        (xyz[:, 1] <= (xyz_min[1] + intervals[1] * (y + 1)))
                    )[0]
                    print(f'{(xyz_min[0] + intervals[0] * x):.4f}<=x<=inf\n'
                          f'{(xyz_min[1] + intervals[1] * y):.4f}<=y<={(xyz_min[1] + intervals[1] * (y + 1)):.4f}')
                elif y == num_splits - 1:
                    grid[x][y] = np.where(
                        (xyz[:, 0] >= (xyz_min[0] + intervals[0] * x)) &
                        (xyz[:, 0] <= (xyz_min[0] + intervals[0] * (x + 1))) &
                        (xyz[:, 1] >= (xyz_min[1] + intervals[1] * y))
                        # (xyz[:, 1] <= (xyz_min[1] + intervals[1] * (y + 1)))
                    )[0]
                    print(f'{(xyz_min[0] + intervals[0] * x):.4f}<=x<={(xyz_min[0] + intervals[0] * (x + 1)):.4f}\n'
                          f'{(xyz_min[1] + intervals[1] * y):.4f}<=y<=inf')
                else:
                    grid[x][y] = np.where(
                        (xyz[:, 0] >= (xyz_min[0] + intervals[0] * x)) &
                        (xyz[:, 0] <= (xyz_min[0] + intervals[0] * (x + 1))) &
                        (xyz[:, 1] >= (xyz_min[1] + intervals[1] * y)) &
                        (xyz[:, 1] <= (xyz_min[1] + intervals[1] * (y + 1)))
                    )[0]
                    print(f'{(xyz_min[0] + intervals[0] * x):.4f}<=x<={(xyz_min[0] + intervals[0] * (x + 1)):.4f}\n'
                          f'{(xyz_min[1] + intervals[1] * y):.4f}<=y<={(xyz_min[1] + intervals[1] * (y + 1)):.4f}')
                total += len(grid[x][y])
                print(f'points = {len(grid[x][y])}\n')

        assert total == len(xyz), "Change the grid splitting code in DataProcessing"
        segments = flatten_list(grid)
        # label_segments = [rgb[s, 0] for s in segments]
        # xyz_segments = [xyz[s] for s in segments]

        print("DEBUG: Grid cell totals")
        for col in reversed(range(len(grid))):
            for row in range(len(grid[col])):
                print(str(len(grid[row][col])).ljust(8), end='\t')
            print()

        grid_mask = np.zeros(len(xyz))
        for val, s in enumerate(segments):
            grid_mask[s] = val

        import pptk
        v = pptk.viewer(xyz, grid_mask, rgb[:, 0])

        sorted_idxs = grid_mask.argsort()
        grid_mask = grid_mask[sorted_idxs]
        return convert_to_pointcloud(xyz[sorted_idxs], intensity[sorted_idxs], rgb[sorted_idxs]), np.split(grid_mask, np.cumsum([len(s) for s in segments]))[:-1]

    chunk_sizes = [len(s) for s in label_segments]
    discard_points = [np.sum(s) for s in label_segments]
    x_distances = [seg[:, 0].max() - seg[:, 0].min() for seg in xyz_segments if seg.size > 0]
    y_distances = [seg[:, 1].max() - seg[:, 1].min() for seg in xyz_segments if seg.size > 0]
    areas = [(seg[:, 1].max() - seg[:, 1].min()) * (seg[:, 0].max() - seg[:, 0].min()) for seg in xyz_segments if
             seg.size > 0]
    print(
        f"Split {len(rgb)} points along the x axis into {num_splits if segment_method != 'grid' else num_splits * num_splits} chunks of size:\n"
        f"{chunk_sizes}\n"
        f"Num 'discard' points per chunk:\n"
        f"{discard_points}\n"
        f"Percentage removed points per chunk:\n"
        f"{[x * 100 / y for x, y in zip(discard_points, chunk_sizes)]}\n"
        f"x-distance in each chunk:\n"
        f"{x_distances}\n"
        f"y-distance in each chunk:\n"
        f"{y_distances}\n"
        f"Area of each chunk:\n"
        f"{areas}")

    # TODO This is 2813036 points (10%)
    # v_chunk = pptk.viewer(xyz[(4943053*2):(4943053*3)], rgb[(4943053*2):(4943053*3)])
    segments_old = [len(label_segments)]
    for i in range(len(label_segments)):
        # segments_old.append(np.copy(segments[i]))
        label_segments[i].fill(i)

    return convert_to_pointcloud(xyz, intensity, rgb), label_segments


def sample_pointcloud(pcd, num_points, segments=None):
    if segments is not None:
        assert num_points < len(segments[-1]), "Too many points for this pointcloud"
    else:
        assert num_points < len(pcd.points), "Too many points for this pointcloud"

    selection_mask = np.zeros(len(pcd.points))
    selection_ids = [np.random.randint(low=0, high=segments[i].size, size=num_points) for i in range(len(segments))]
    selection_mask[flatten_list([si * (num + 1) for num, si in enumerate(selection_ids)])] = 1

    return selection_ids, selection_mask


def sort_pointcloud(pointcloud, axis='x'):
    """
    Sorts the pointcloud (possibly in place??) by the given axis
    :param pointcloud: open3d pointcloud
    :param axis: x/y/z to sort along
    :return: the sorted ndarrays xyz, intensity, rgb
    """
    assert axis in ['x', 'y', 'z'], "Must sort by x,y,z axis"
    print(f"Sorting points by {axis}-coordinate")
    axis = axis_dict[axis]
    xyz, intensity, rgb = convert_to_arrays(pointcloud)
    axis_sorted_indices = xyz[:, axis].argsort()  # get the indices for xyz sorted on x
    return xyz[axis_sorted_indices], intensity[axis_sorted_indices], rgb[axis_sorted_indices]


def generate_segment_mask(size, splits):
    return [x // size for x in range(size)]


def convert_to_arrays(pointcloud):
    return np.asarray(pointcloud.points), np.asarray(pointcloud.normals)[:, 0], np.asarray(pointcloud.colors)


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
