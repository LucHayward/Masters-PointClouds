import DataProcessing
from pathlib import Path
import numpy as np
from numpy.random import default_rng
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser('generate')
parser.add_argument('--dataset', default='Church', choices=['Church', 'SongMnara'])
args = parser.parse_args()

rng = default_rng()

area_map = {'Church': 'Area_1',
            'SongoMnara': 'Area_2'}

# church_file = Path('Data/Church/Church.ply')
church_file = Path('../../PatrickData/Church/Church.ply')
# songo_mnara_file = Path('Data/SongoMnara/SongoMnara.ply')
songo_mnara_file = Path('../../PatrickData/SongoMnara/SongoMnara.ply')
# songo_mnara_uds5_file = Path('Data/SongoMnara/SongoMnara_uds5.ply')
songo_mnara_uds5_file = Path('../../PatrickData/SongoMnara/SongoMnara_uds5.ply')
s3dis_data_format_dir = Path('../../PatrickData/Church/s3disFormat')

pointcloud = DataProcessing.load_from_ply(church_file)

pointcloud, segments = DataProcessing.segment_pointcloud(pointcloud, 2, segment_method='grid', sort_axis='x')

# Need it in the format
# Area_x/Segment_x/Annotations/
# Area_x/Segment_x/Annotations/keep.txt
# Area_x/Segment_x/Annotations/discard.txt
# Area_x/Segment_x/Segment_x.txt
#


# TODO implement this better (spread center data around) and add to dataprocessing
#  changed to using 1024 because I want dinner
# TODO ensure that all segments have at least 1024 points, can manage this by addressing keep and discard separately
#  For now have just left it as uniform because I'm hungry and its 10pm
segment_sizes = [x.size for x in segments]

ss = []
segs = []
temp_seg = None
last_merge = 0
temp = 0
# minimum_segment_size = np.median(segment_sizes)
minimum_segment_size = 4096
for i, size in tqdm(enumerate(segment_sizes)):
    if size < minimum_segment_size:
        temp += size
        if temp_seg is None:
            temp_seg = segments[i]
        else:
            temp_seg = np.concatenate((temp_seg, segments[i]))
    elif temp != 0 and (temp + size) >= minimum_segment_size:
        ss.append(temp + size)
        temp = 0
        segs.append(np.concatenate((temp_seg, segments[i])))
        temp_seg = None
    else:
        ss.append(size)
        segs.append(segments[i])
    if temp >= minimum_segment_size:
        ss.append(temp)
        segs.append(temp_seg)
        temp_seg = None
        temp = 0
if temp_seg is not None:
    if temp_seg.size < minimum_segment_size:
        ss[-1] += temp
        segs[-1] = np.concatenate((segs[-1], temp_seg))
    segs.append(temp_seg)
    ss.append(temp)
segments = segs
segment_sizes = [x.size for x in segments]

xyz, intensity, rgb = DataProcessing.convert_to_arrays(pointcloud)
# class_sorted_indices = rgb[:, 0].argsort()
# xyz, intensity, rgb = xyz[class_sorted_indices], intensity[class_sorted_indices], rgb[class_sorted_indices]
print(f'Num Total points: {rgb.size}\nNum Total Discard points: {rgb[:, 0].sum()} ')
discarded_points = []
cnt = 0
for seg in segments:
    discarded_points.append(np.sum(rgb[cnt:cnt + len(seg), 0]))
    cnt += len(seg)
print(f"Discarded points per segment:\n{discarded_points}")

discard_cnt = 0
point_id = 0
# TODO remove all files in folders below before running
# TODO Allow for single scan to = multiplea areas
# Write the data out pointwise into keep/discard files and replace rgb with IntensityGB
meta_dir = s3dis_data_format_dir.joinpath(f'../meta_{args.dataset}'.lower())
meta_dir.mkdir(exist_ok=True, parents=True)
with open(meta_dir.joinpath('anno_paths.txt'), 'w+') as anno_paths_outfile, \
        open(meta_dir.joinpath('class_names.txt'), 'w+') as class_names_outfile:
    class_names_outfile.write('keep\ndiscard')

    for segment_id in tqdm(range(len(segments))):
        segment_dir = s3dis_data_format_dir.joinpath(area_map.get(args.dataset), f'segment_{segment_id + 1}')
        annotations_dir = segment_dir.joinpath('Annotations/')
        annotations_dir.mkdir(exist_ok=True, parents=True)

        with open(annotations_dir.joinpath('keep_1.txt'), 'w+') as keep_outfile, \
                open(annotations_dir.joinpath('discard_1.txt'), 'w+') as discard_outfile, \
                open(segment_dir.joinpath(f'segment_{segment_id + 1}.txt'), 'w+') as points_outfile:

            anno_paths_outfile.write(f'{area_map.get(args.dataset)}/segment_{segment_id + 1}/Annotations\n')

            for point in xyz[point_id:point_id + segments[segment_id].size - 1]:
                point_id += 1
                # Multiply IGB data by 255 as s3dis expects prenormalised input
                points_outfile.write(
                    f'{point[0]} {point[1]} {point[2]} {intensity[point_id] * 255} {rgb[point_id, 1] * 255} {rgb[point_id, 2] * 255}\n')
                if rgb[point_id, 0] == 0:
                    keep_outfile.write(
                        f'{point[0]} {point[1]} {point[2]} {intensity[point_id] * 255} {rgb[point_id, 1] * 255} {rgb[point_id, 2 * 255]}\n')
                else:
                    discard_cnt += 1
                    discard_outfile.write(
                        f'{point[0]} {point[1]} {point[2]} {intensity[point_id] * 255} {rgb[point_id, 1] * 255} {rgb[point_id, 2] * 255}\n')

print(f'Discard count = {discard_cnt}')

print("Done")
