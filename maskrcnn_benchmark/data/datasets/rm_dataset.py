# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Simple dataset class that wraps a custom dataset
[
    {
        'filename': 'a.jpg',
        'width': 1280,
        'height': 720,
        'ann': {
            'bboxes': <np.ndarray> (n, 4),
            'labels': <np.ndarray> (n, ),
            'bboxes_ignore': <np.ndarray> (k, 4),
            'labels_ignore': <np.ndarray> (k, 4) (optional field)
        }
    },
    ...
]
"""
import os
from PIL import Image
import pickle as pkl
import torch

from maskrcnn_benchmark.structures.bounding_box import BoxList


class RMDataset(object):
    def __init__(self, root, ann_file, transforms=None):
        with open(ann_file, 'rb') as fid:
            self.anno_lists = pkl.load(fid)
        self.root = root
        self.transforms = transforms
        self.classes = ['__background__',  # always index 0
                        'c60', 'c70', 'car_people', 'center_ring', 'cross_hatch',
                        'diamond', 'forward_left', 'forward_right', 'forward',
                        'left', 'right', 'u_turn', 'zebra_crossing']
        self.num_classes = len(self.classes)

    def __getitem__(self, item):
        img_anno = self.anno_lists[item]
        img = Image.open(os.path.join(self.root,img_anno['filename'])).convert("RGB")

        boxes = img_anno['ann']['bboxes']
        labels = img_anno['ann']['labels']
        target = BoxList(boxes, img.size, mode="xyxy")
        target.add_field("labels", torch.tensor(labels))

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, item

    def __len__(self):
        return len(self.anno_lists)

    def get_img_info(self, item):
        """
        Return the image dimensions for the image, without
        loading and pre-processing it
        """
        img_anno = self.anno_lists[item]
        return {'height':img_anno['height'],'width':img_anno['width']}

if __name__ == '__main__':
    import pdb
    root = '/home/opt48/zhangxq/github/maskrcnn-benchmark/datasets/voc'
    rmdataset = RMDataset(root,'vocRM_train.pkl')
    print(len(rmdataset))
    for img, target, item in rmdataset:
        print(img)
        print(target)
        print(rmdataset.get_img_info(item))
        pdb.set_trace()