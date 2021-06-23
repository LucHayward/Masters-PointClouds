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
    turbo_colormap_data = [[0.18995, 0.07176, 0.23217], [0.19483, 0.08339, 0.26149], [0.19956, 0.09498, 0.29024],
                           [0.20415, 0.10652, 0.31844], [0.20860, 0.11802, 0.34607], [0.21291, 0.12947, 0.37314],
                           [0.21708, 0.14087, 0.39964], [0.22111, 0.15223, 0.42558], [0.22500, 0.16354, 0.45096],
                           [0.22875, 0.17481, 0.47578], [0.23236, 0.18603, 0.50004], [0.23582, 0.19720, 0.52373],
                           [0.23915, 0.20833, 0.54686], [0.24234, 0.21941, 0.56942], [0.24539, 0.23044, 0.59142],
                           [0.24830, 0.24143, 0.61286], [0.25107, 0.25237, 0.63374], [0.25369, 0.26327, 0.65406],
                           [0.25618, 0.27412, 0.67381], [0.25853, 0.28492, 0.69300], [0.26074, 0.29568, 0.71162],
                           [0.26280, 0.30639, 0.72968], [0.26473, 0.31706, 0.74718], [0.26652, 0.32768, 0.76412],
                           [0.26816, 0.33825, 0.78050], [0.26967, 0.34878, 0.79631], [0.27103, 0.35926, 0.81156],
                           [0.27226, 0.36970, 0.82624], [0.27334, 0.38008, 0.84037], [0.27429, 0.39043, 0.85393],
                           [0.27509, 0.40072, 0.86692], [0.27576, 0.41097, 0.87936], [0.27628, 0.42118, 0.89123],
                           [0.27667, 0.43134, 0.90254], [0.27691, 0.44145, 0.91328], [0.27701, 0.45152, 0.92347],
                           [0.27698, 0.46153, 0.93309], [0.27680, 0.47151, 0.94214], [0.27648, 0.48144, 0.95064],
                           [0.27603, 0.49132, 0.95857], [0.27543, 0.50115, 0.96594], [0.27469, 0.51094, 0.97275],
                           [0.27381, 0.52069, 0.97899], [0.27273, 0.53040, 0.98461], [0.27106, 0.54015, 0.98930],
                           [0.26878, 0.54995, 0.99303], [0.26592, 0.55979, 0.99583], [0.26252, 0.56967, 0.99773],
                           [0.25862, 0.57958, 0.99876], [0.25425, 0.58950, 0.99896], [0.24946, 0.59943, 0.99835],
                           [0.24427, 0.60937, 0.99697], [0.23874, 0.61931, 0.99485], [0.23288, 0.62923, 0.99202],
                           [0.22676, 0.63913, 0.98851], [0.22039, 0.64901, 0.98436], [0.21382, 0.65886, 0.97959],
                           [0.20708, 0.66866, 0.97423], [0.20021, 0.67842, 0.96833], [0.19326, 0.68812, 0.96190],
                           [0.18625, 0.69775, 0.95498], [0.17923, 0.70732, 0.94761], [0.17223, 0.71680, 0.93981],
                           [0.16529, 0.72620, 0.93161], [0.15844, 0.73551, 0.92305], [0.15173, 0.74472, 0.91416],
                           [0.14519, 0.75381, 0.90496], [0.13886, 0.76279, 0.89550], [0.13278, 0.77165, 0.88580],
                           [0.12698, 0.78037, 0.87590], [0.12151, 0.78896, 0.86581], [0.11639, 0.79740, 0.85559],
                           [0.11167, 0.80569, 0.84525], [0.10738, 0.81381, 0.83484], [0.10357, 0.82177, 0.82437],
                           [0.10026, 0.82955, 0.81389], [0.09750, 0.83714, 0.80342], [0.09532, 0.84455, 0.79299],
                           [0.09377, 0.85175, 0.78264], [0.09287, 0.85875, 0.77240], [0.09267, 0.86554, 0.76230],
                           [0.09320, 0.87211, 0.75237], [0.09451, 0.87844, 0.74265], [0.09662, 0.88454, 0.73316],
                           [0.09958, 0.89040, 0.72393], [0.10342, 0.89600, 0.71500], [0.10815, 0.90142, 0.70599],
                           [0.11374, 0.90673, 0.69651], [0.12014, 0.91193, 0.68660], [0.12733, 0.91701, 0.67627],
                           [0.13526, 0.92197, 0.66556], [0.14391, 0.92680, 0.65448], [0.15323, 0.93151, 0.64308],
                           [0.16319, 0.93609, 0.63137], [0.17377, 0.94053, 0.61938], [0.18491, 0.94484, 0.60713],
                           [0.19659, 0.94901, 0.59466], [0.20877, 0.95304, 0.58199], [0.22142, 0.95692, 0.56914],
                           [0.23449, 0.96065, 0.55614], [0.24797, 0.96423, 0.54303], [0.26180, 0.96765, 0.52981],
                           [0.27597, 0.97092, 0.51653], [0.29042, 0.97403, 0.50321], [0.30513, 0.97697, 0.48987],
                           [0.32006, 0.97974, 0.47654], [0.33517, 0.98234, 0.46325], [0.35043, 0.98477, 0.45002],
                           [0.36581, 0.98702, 0.43688], [0.38127, 0.98909, 0.42386], [0.39678, 0.99098, 0.41098],
                           [0.41229, 0.99268, 0.39826], [0.42778, 0.99419, 0.38575], [0.44321, 0.99551, 0.37345],
                           [0.45854, 0.99663, 0.36140], [0.47375, 0.99755, 0.34963], [0.48879, 0.99828, 0.33816],
                           [0.50362, 0.99879, 0.32701], [0.51822, 0.99910, 0.31622], [0.53255, 0.99919, 0.30581],
                           [0.54658, 0.99907, 0.29581], [0.56026, 0.99873, 0.28623], [0.57357, 0.99817, 0.27712],
                           [0.58646, 0.99739, 0.26849], [0.59891, 0.99638, 0.26038], [0.61088, 0.99514, 0.25280],
                           [0.62233, 0.99366, 0.24579], [0.63323, 0.99195, 0.23937], [0.64362, 0.98999, 0.23356],
                           [0.65394, 0.98775, 0.22835], [0.66428, 0.98524, 0.22370], [0.67462, 0.98246, 0.21960],
                           [0.68494, 0.97941, 0.21602], [0.69525, 0.97610, 0.21294], [0.70553, 0.97255, 0.21032],
                           [0.71577, 0.96875, 0.20815], [0.72596, 0.96470, 0.20640], [0.73610, 0.96043, 0.20504],
                           [0.74617, 0.95593, 0.20406], [0.75617, 0.95121, 0.20343], [0.76608, 0.94627, 0.20311],
                           [0.77591, 0.94113, 0.20310], [0.78563, 0.93579, 0.20336], [0.79524, 0.93025, 0.20386],
                           [0.80473, 0.92452, 0.20459], [0.81410, 0.91861, 0.20552], [0.82333, 0.91253, 0.20663],
                           [0.83241, 0.90627, 0.20788], [0.84133, 0.89986, 0.20926], [0.85010, 0.89328, 0.21074],
                           [0.85868, 0.88655, 0.21230], [0.86709, 0.87968, 0.21391], [0.87530, 0.87267, 0.21555],
                           [0.88331, 0.86553, 0.21719], [0.89112, 0.85826, 0.21880], [0.89870, 0.85087, 0.22038],
                           [0.90605, 0.84337, 0.22188], [0.91317, 0.83576, 0.22328], [0.92004, 0.82806, 0.22456],
                           [0.92666, 0.82025, 0.22570], [0.93301, 0.81236, 0.22667], [0.93909, 0.80439, 0.22744],
                           [0.94489, 0.79634, 0.22800], [0.95039, 0.78823, 0.22831], [0.95560, 0.78005, 0.22836],
                           [0.96049, 0.77181, 0.22811], [0.96507, 0.76352, 0.22754], [0.96931, 0.75519, 0.22663],
                           [0.97323, 0.74682, 0.22536], [0.97679, 0.73842, 0.22369], [0.98000, 0.73000, 0.22161],
                           [0.98289, 0.72140, 0.21918], [0.98549, 0.71250, 0.21650], [0.98781, 0.70330, 0.21358],
                           [0.98986, 0.69382, 0.21043], [0.99163, 0.68408, 0.20706], [0.99314, 0.67408, 0.20348],
                           [0.99438, 0.66386, 0.19971], [0.99535, 0.65341, 0.19577], [0.99607, 0.64277, 0.19165],
                           [0.99654, 0.63193, 0.18738], [0.99675, 0.62093, 0.18297], [0.99672, 0.60977, 0.17842],
                           [0.99644, 0.59846, 0.17376], [0.99593, 0.58703, 0.16899], [0.99517, 0.57549, 0.16412],
                           [0.99419, 0.56386, 0.15918], [0.99297, 0.55214, 0.15417], [0.99153, 0.54036, 0.14910],
                           [0.98987, 0.52854, 0.14398], [0.98799, 0.51667, 0.13883], [0.98590, 0.50479, 0.13367],
                           [0.98360, 0.49291, 0.12849], [0.98108, 0.48104, 0.12332], [0.97837, 0.46920, 0.11817],
                           [0.97545, 0.45740, 0.11305], [0.97234, 0.44565, 0.10797], [0.96904, 0.43399, 0.10294],
                           [0.96555, 0.42241, 0.09798], [0.96187, 0.41093, 0.09310], [0.95801, 0.39958, 0.08831],
                           [0.95398, 0.38836, 0.08362], [0.94977, 0.37729, 0.07905], [0.94538, 0.36638, 0.07461],
                           [0.94084, 0.35566, 0.07031], [0.93612, 0.34513, 0.06616], [0.93125, 0.33482, 0.06218],
                           [0.92623, 0.32473, 0.05837], [0.92105, 0.31489, 0.05475], [0.91572, 0.30530, 0.05134],
                           [0.91024, 0.29599, 0.04814], [0.90463, 0.28696, 0.04516], [0.89888, 0.27824, 0.04243],
                           [0.89298, 0.26981, 0.03993], [0.88691, 0.26152, 0.03753], [0.88066, 0.25334, 0.03521],
                           [0.87422, 0.24526, 0.03297], [0.86760, 0.23730, 0.03082], [0.86079, 0.22945, 0.02875],
                           [0.85380, 0.22170, 0.02677], [0.84662, 0.21407, 0.02487], [0.83926, 0.20654, 0.02305],
                           [0.83172, 0.19912, 0.02131], [0.82399, 0.19182, 0.01966], [0.81608, 0.18462, 0.01809],
                           [0.80799, 0.17753, 0.01660], [0.79971, 0.17055, 0.01520], [0.79125, 0.16368, 0.01387],
                           [0.78260, 0.15693, 0.01264], [0.77377, 0.15028, 0.01148], [0.76476, 0.14374, 0.01041],
                           [0.75556, 0.13731, 0.00942], [0.74617, 0.13098, 0.00851], [0.73661, 0.12477, 0.00769],
                           [0.72686, 0.11867, 0.00695], [0.71692, 0.11268, 0.00629], [0.70680, 0.10680, 0.00571],
                           [0.69650, 0.10102, 0.00522], [0.68602, 0.09536, 0.00481], [0.67535, 0.08980, 0.00449],
                           [0.66449, 0.08436, 0.00424], [0.65345, 0.07902, 0.00408], [0.64223, 0.07380, 0.00401],
                           [0.63082, 0.06868, 0.00401], [0.61923, 0.06367, 0.00410], [0.60746, 0.05878, 0.00427],
                           [0.59550, 0.05399, 0.00453], [0.58336, 0.04931, 0.00486], [0.57103, 0.04474, 0.00529],
                           [0.55852, 0.04028, 0.00579], [0.54583, 0.03593, 0.00638], [0.53295, 0.03169, 0.00705],
                           [0.51989, 0.02756, 0.00780], [0.50664, 0.02354, 0.00863], [0.49321, 0.01963, 0.00955],
                           [0.47960, 0.01583, 0.01055]]

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
        intervals = total_distances / num_splits
        # xs = np.arange(pointcloud.get_min_bound()[0], pointcloud.get_max_bound()[0], intervals[0])
        # ys = np.arange(pointcloud.get_min_bound()[1], pointcloud.get_max_bound()[1], intervals[1])
        xs = np.linspace(pointcloud.get_min_bound()[0], pointcloud.get_max_bound()[0], num_splits+1)
        ys = np.linspace(pointcloud.get_min_bound()[1], pointcloud.get_max_bound()[1], num_splits+1)
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
                        (xyz[:, 1] < (xyz_min[1] + intervals[1] * (y + 1)))
                    )[0]
                    # print(f'{(xyz_min[0] + intervals[0] * x):.4f}<=x<=inf\n'
                    #       f'{(xyz_min[1] + intervals[1] * y):.4f}<=y<={(xyz_min[1] + intervals[1] * (y + 1)):.4f}')
                elif y == num_splits - 1:
                    grid[x][y] = np.where(
                        (xyz[:, 0] >= (xyz_min[0] + intervals[0] * x)) &
                        (xyz[:, 0] < (xyz_min[0] + intervals[0] * (x + 1))) &
                        (xyz[:, 1] >= (xyz_min[1] + intervals[1] * y))
                        # (xyz[:, 1] <= (xyz_min[1] + intervals[1] * (y + 1)))
                    )[0]
                    # print(f'{(xyz_min[0] + intervals[0] * x):.4f}<=x<={(xyz_min[0] + intervals[0] * (x + 1)):.4f}\n'
                    #       f'{(xyz_min[1] + intervals[1] * y):.4f}<=y<=inf')
                else:
                    grid[x][y] = np.where(
                        (xyz[:, 0] >= (xyz_min[0] + intervals[0] * x)) &
                        (xyz[:, 0] < (xyz_min[0] + intervals[0] * (x + 1))) &
                        (xyz[:, 1] >= (xyz_min[1] + intervals[1] * y)) &
                        (xyz[:, 1] < (xyz_min[1] + intervals[1] * (y + 1)))
                    )[0]
                    # print(f'{(xyz_min[0] + intervals[0] * x):.4f}<=x<={(xyz_min[0] + intervals[0] * (x + 1)):.4f}\n'
                    #       f'{(xyz_min[1] + intervals[1] * y):.4f}<=y<={(xyz_min[1] + intervals[1] * (y + 1)):.4f}')
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
        for i,v in enumerate(np.unique(grid_mask)):
            grid_mask[grid_mask == v] = i

        import pptk
        v = pptk.viewer(xyz, grid_mask, rgb[:, 0])
        v.color_map(turbo_colormap_data)
        # sorted_idxs = grid_mask.argsort()

        # return convert_to_pointcloud(xyz[sorted_idxs], intensity[sorted_idxs], rgb[sorted_idxs]), np.split(grid_mask, np.cumsum([len(s) for s in segments]))[:-1]
        return convert_to_pointcloud(xyz, intensity, rgb), segments

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
