import DataProcessing
from pathlib import Path
import numpy as np
from numpy.random import default_rng

rng = default_rng()

# church_file = Path('Data/Church/Church.ply')
church_file = Path('../../PatrickData/Church/Church.ply')
# songo_mnara_file = Path('Data/SongoMnara/SongoMnara.ply')
songo_mnara_file = Path('../../PatrickData/SongoMnara/SongoMnara.ply')
# songo_mnara_uds5_file = Path('Data/SongoMnara/SongoMnara_uds5.ply')
songo_mnara_uds5_file = Path('../../PatrickData/SongoMnara/SongoMnara_uds5.ply')
modelnet_data_format_dir = Path('../../PatrickData/Church/ModelNetFormat')

pointcloud = DataProcessing.load_from_ply(church_file)

pointcloud, segments = DataProcessing.segment_pointcloud(pointcloud, 50, segment_method='spatial', sort_axis='x')

# TODO implement this better (spread center data around) and add to dataprocessing
# TODO changed to using 1024 because I want dinner
segment_sizes = [x.size for x in segments]

ss = []
segs = []
temp_seg = None
last_merge = 0
temp = 0
for i, size in enumerate(segment_sizes):
    if size < np.median(segment_sizes):
        temp += size
        if temp_seg is None:
            temp_seg = segments[i]
        else:
            temp_seg = np.concatenate((temp_seg, segments[i]))
    elif temp != 0 and (temp+size) >= np.median(segment_sizes):
        ss.append(temp+size)
        temp = 0
        segs.append(np.concatenate((temp_seg, segments[i])))
        temp_seg = None
    else:
        ss.append(size)
        segs.append(segments[i])
    if temp >= np.median(segment_sizes):
        ss.append(temp)
        segs.append(temp_seg)
        temp_seg = None
        temp = 0
if temp_seg is not None:
    segs.append(temp_seg)
    ss.append(temp)
segments = segs
segment_sizes = [x.size for x in segments]

xyz, intensity, rgb = DataProcessing.convert_to_arrays(pointcloud)
# class_sorted_indices = rgb[:, 0].argsort()
# xyz, intensity, rgb = xyz[class_sorted_indices], intensity[class_sorted_indices], rgb[class_sorted_indices]
print(f'Num Discard points: {rgb[:, 0].sum()}')
discard_cnt = 0
point_id = 0

# TODO remove all files in folders below before running

for segment_id in range(len(segments)):
    with open(modelnet_data_format_dir.joinpath(f'keep/keep_{str(segment_id + 1).zfill(4)}.txt'), 'w+') as keep_outfile, \
            open(modelnet_data_format_dir.joinpath(f'discard/discard_{str(segment_id + 1).zfill(4)}.txt'),
                 'w+') as discard_outfile:
        for point in xyz[point_id:point_id+segments[segment_id].size-1]:
            point_id += 1
            if rgb[point_id, 0] == 0:
                keep_outfile.write(f'{point[0]},{point[1]},{point[2]},0,0,0\n')
            else:
                discard_cnt += 1
                discard_outfile.write(f'{point[0]},{point[1]},{point[2]},0,0,0\n')

print(f'Discard count = {discard_cnt}')

with open(modelnet_data_format_dir.joinpath('church_train.txt'), 'w+') as church_train_file, \
        open(modelnet_data_format_dir.joinpath('church_test.txt'), 'w+') as church_test_file:
    #     TODO Fix the choice of test categories
    gt_median = [x > np.median(segment_sizes) for x in segment_sizes]
    # Do generate len(sizes)//5 for 5-fold cross val enabling (also in keeping with modelnet)
    # test_file_nums = rng.integers(gt_median.index(True), len(sizes) - gt_median[::-1].index(True),
    #                               len(segments) // 5)
    test_file_nums = rng.choice(np.arange(gt_median.index(True), len(segment_sizes) - gt_median[::-1].index(True)),
                                len(segment_sizes) // 5, replace=False, shuffle=False)
    # test_file_nums += 1

    # test_file_nums.sort()

    for segment_id in range(len(segments)):
        if segment_id + 1 in test_file_nums:
            church_test_file.write(f'keep_{str(segment_id + 1).zfill(4)}\n')
        else:
            church_train_file.write(f'keep_{str(segment_id + 1).zfill(4)}\n')
    for segment_id in range(len(segments)):
        if segment_id in test_file_nums:
            church_test_file.write(f'discard_{str(segment_id + 1).zfill(4)}\n')
        else:
            church_train_file.write(f'discard_{str(segment_id + 1).zfill(4)}\n')

print("Done")
