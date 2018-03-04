"""
Tests whether the serializers work properly
"""
import os

import numpy as np

from noxer.serializers import FolderDatasetReader


class TestSerializers:
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_folder_dataset(self):
        folder = os.path.join('test_data', 'folder_dataset')
        dataset = FolderDatasetReader(folder)

        X, Y = dataset.read(-1, 'train')

        # check if labels are there
        categories = ['class img', 'class_img_2', 'class_json', 'class_wav']

        for c in categories:
            if not c in Y:
                raise ValueError("The category %s was not read!" % c)

        # check if data is read corrrectly
        for x, y in zip(X, Y):
            if y == "class img":
                assert x.shape == (32, 32, 3)
                assert np.mean(x) == 228.400390625
            if y == "class_img_2":
                assert x.shape == (26, 32, 3)
                assert np.mean(x) == 196.75480769230768
            if y == "class_wav":
                assert x.shape == (1, 41294)
                assert np.mean(x)*1e+6 == 67.19572411384434