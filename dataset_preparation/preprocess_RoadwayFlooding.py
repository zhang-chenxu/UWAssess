import os

dataset_path = '../test_set/Dataset'
target_path = '../test_set/RoadwayFlooding'

for label_name in os.listdir(os.path.join(dataset_path, 'labels')):
    new_name = label_name.replace('label', 'image')
    os.rename(os.path.join(dataset_path, 'labels', label_name), os.path.join(dataset_path, 'labels', new_name))

os.rename(os.path.join(dataset_path, 'labels'), os.path.join(dataset_path, 'SegmentationClass'))
os.rename(os.path.join(dataset_path, 'images'), os.path.join(dataset_path, 'JPEGImages'))
os.rename(dataset_path, target_path)
