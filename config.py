import os

data_dir = r'F:\DataSet\NYU_Depth2'
mat_path = os.path.join(data_dir, r'nyu_depth_v2_labeled.mat')

working_path = os.path.abspath(os.path.dirname(__file__))

coco_dir = r'F:\DataSet\COCO'


weights_dir = os.path.join(working_path, 'weights')
test_dir = os.path.join(working_path, 'test_images')
