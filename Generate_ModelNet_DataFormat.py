import pptk
import DataProcessing
from pathlib import Path
import numpy as np
import open3d as o3d

# church_file = Path('Data/Church/Church.ply')
church_file = Path('../../PatrickData/Church/Church.ply')
# songo_mnara_file = Path('Data/SongoMnara/SongoMnara.ply')
songo_mnara_file = Path('../../PatrickData/SongoMnara/SongoMnara.ply')
# songo_mnara_uds5_file = Path('Data/SongoMnara/SongoMnara_uds5.ply')
songo_mnara_uds5_file = Path('../../PatrickData/SongoMnara/SongoMnara_uds5.ply')
modelnet_data_format_dir = Path('../../PatrickData/Church/ModelNetFormat')

pointcloud = DataProcessing.load_from_ply(church_file)

pointcloud, segments = DataProcessing.segment_pointcloud(pointcloud, 50, segment_method='spatial', sort_axis='x')

xyz, intensity, rgb = DataProcessing.convert_to_arrays(pointcloud)
class_sorted_indices = rgb[:, 0].argsort()
xyz, intensity, rgb = xyz[class_sorted_indices], intensity[class_sorted_indices], rgb[class_sorted_indices]
print(f'Num Discard points: {rgb[:,0].sum()}')
discard_cnt = 0
point_id = 0

for segment_id in range(len(segments)):
    with open(modelnet_data_format_dir.joinpath(f'keep/keep_{str(segment_id+1).zfill(4)}.txt'), 'w+') as keep_outfile, \
            open(modelnet_data_format_dir.joinpath(f'discard/discard_{str(segment_id+1).zfill(4)}.txt'), 'w+') as discard_outfile:
        for point in xyz[point_id:segments[segment_id].size]:
            point_id += 1
            if rgb[point_id, 0] == 0:
                keep_outfile.write(f'{point[0]},{point[1]},{point[2]},0,0,0\n')
            else:
                discard_cnt += 1
                discard_outfile.write(f'{point[0]},{point[1]},{point[2]},0,0,0\n')

print(f'Discard count = {discard_cnt}')

with open(modelnet_data_format_dir.joinpath('church_train.txt'), 'w+') as church_train_file, \
        open(modelnet_data_format_dir.joinpath('church_test.txt'), 'w+') as church_test_file:
    sizes = [x.size for x in segments]
    #     TODO Fix the choice of test categories
    gt_median = [x > np.median(sizes) for x in sizes]
    # Do generate len(sizes)//5 for 5-fold cross val enabling (also in keeping with modelnet)
    test_file_nums = np.random.randint(gt_median.index(True), len(sizes) - gt_median[::-1].index(True),
                                       len(segments) // 5)

    last_segment_tested = -1
    for segment_id in range(len(segments)):
        if segment_id in test_file_nums:
            church_test_file.write(f'keep_{str(segment_id+1).zfill(4)}\n')
        else:
            church_train_file.write(f'keep_{str(segment_id+1).zfill(4)}\n')
    for segment_id in range(len(segments)):
        if segment_id in test_file_nums:
            church_test_file.write(f'discard_{str(segment_id+1).zfill(4)}\n')
        else:
            church_train_file.write(f'discard_{str(segment_id+1).zfill(4)}\n')

print("Done")
