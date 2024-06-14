# 
# Modified from:
#   https://github.com/tangjiapeng/DiffuScene.
# 

import numpy as np
from torch.utils.data import dataloader

from threed_front.datasets.threed_front_encoding_base import *
from threed_front.datasets import get_raw_dataset


class Diffusion(DatasetDecoratorBase):
    def __init__(self, dataset, max_length=None):
        super().__init__(dataset)
        
        if max_length is None:
            self._max_length = dataset.max_length
        else:
            assert max_length >= dataset.max_length
            self._max_length = max_length
    
    def __getitem__(self, idx):
        sample_params = self._dataset[idx]

        # Add the number of bounding boxes in the scene
        sample_params["length"] = sample_params["class_labels"].shape[0]
        
        sample_params_target = {}
        # Compute the target from the input
        for k, v in sample_params.items():
            if k in [
                "room_layout", "length", "fpbpn"
            ]:
                pass

            elif k == "class_labels":
                if self._dataset.n_classes == self._dataset.n_object_types + 2:
                    # Delete the 'start' label and keep the last as 'empty' label 
                    class_labels = np.hstack([v[:, :-2], v[:, -1:]])
                else:
                    assert self._dataset.n_classes == self._dataset.n_object_types + 1
                    class_labels = v
                # Pad the 'empty' label in the end of each sequence, 
                # and convert the class labels to -1, 1
                L, C = class_labels.shape
                empty_label = np.eye(C)[-1]
                sample_params_target[k] = np.vstack([
                    class_labels, 
                    np.tile(empty_label[None, :], [self._max_length - L, 1])
                ]).astype(np.float32) * 2.0 - 1.0 

            else:
                # Set the attributes for the 'empty' label
                L, C = v.shape
                sample_params_target[k] = np.vstack([
                    v, np.zeros((self._max_length - L, C))
                ]).astype(np.float32)

        sample_params.update(sample_params_target)

        return sample_params

    @property
    def max_length(self):
        return self._max_length 
    
    def collate_fn(self, samples):
        ''' Collater that puts each data field into a tensor with outer dimension
            batch size.
        Args:
            samples: samples
        '''
    
        samples = list(filter(lambda x: x is not None, samples))
        return dataloader.default_collate(samples)


def get_dataset_raw_and_encoded(
    config,
    filter_fn=lambda s: s,
    path_to_bounds=None,
    augmentations=None,
    split=["train", "val"],
    max_length=None,
    include_room_mask=True,
):
    dataset = get_raw_dataset(
        config, filter_fn, path_to_bounds, split, 
        include_room_mask=include_room_mask
    )
    encoding = dataset_encoding_factory(
        config.get("encoding_type"),
        dataset,
        augmentations,
        config.get("box_ordering", None),
        max_length
    )

    return dataset, encoding


def get_encoded_dataset(
    config,
    filter_fn=lambda s: s,
    path_to_bounds=None,
    augmentations=None,
    split=["train", "val"],
    max_length=None,
    include_room_mask=True
):
    _, encoding = get_dataset_raw_and_encoded(
        config, filter_fn, path_to_bounds, augmentations, split, max_length, 
        include_room_mask
    )
    return encoding

def dataset_encoding_factory(
    name,
    dataset,
    augmentations=None,
    box_ordering=None,
    max_length=None,
):
    # list of object features
    feature_keys = ["class_labels", "translations", "sizes", "angles"]
    if "objfeats" in name:
        if "lat32" in name:
            feature_keys.append("objfeats_32")
            print("use lat32 as objfeats")
        else:
            feature_keys.append("objfeats")
            print("use lat64 as objfeats")
    
    # NOTE: The ordering might change after augmentations so really it should
    #       be done after the augmentations. For class frequencies it is fine
    #       though.
    if "cached" in name:
        dataset_collection = CachedDatasetCollection(dataset)
        if box_ordering:
            dataset_collection = \
                OrderedDataset(dataset_collection, feature_keys, box_ordering)
    else:
        box_ordered_dataset = BoxOrderedDataset(dataset, box_ordering)

        class_labels = ClassLabelsEncoder(box_ordered_dataset)
        translations = TranslationEncoder(box_ordered_dataset)
        sizes = SizeEncoder(box_ordered_dataset)
        angles = AngleEncoder(box_ordered_dataset)
        objfeats = ObjFeatEncoder(box_ordered_dataset)
        objfeats_32 = ObjFeat32Encoder(box_ordered_dataset)

        if name == "basic":
            return DatasetCollection(
                class_labels,
                translations,
                sizes,
                angles,
                objfeats,
                objfeats_32
            )
        
        room_layout = RoomLayoutEncoder(box_ordered_dataset)
        dataset_collection = DatasetCollection(
            room_layout,
            class_labels,
            translations,
            sizes,
            angles,
            objfeats,
            objfeats_32
        )

    if isinstance(augmentations, list):
        for aug_type in augmentations:
            if aug_type == "rotations":
                print("Applying rotation augmentations")
                dataset_collection = RotationAugmentation(dataset_collection)
            elif aug_type == "fixed_rotations":
                print("Applying fixed rotation augmentations")
                dataset_collection = RotationAugmentation(dataset_collection, fixed=True)
            elif aug_type == "jitter":
                print("Applying jittering augmentations")
                dataset_collection = Jitter(dataset_collection)
     
    # Scale the input
    if "cosin_angle" in name:
        dataset_collection = Scale_CosinAngle(dataset_collection)
    else:
        dataset_collection = Scale(dataset_collection)

    # for diffusion (represent objectness as the last channel of class label)
    if "diffusion" in name:
        if "eval" in name:
            return Diffusion(dataset_collection, max_length)
        elif "wocm_no_prm" in name:
            return Diffusion(dataset_collection, max_length)
        elif "wocm" in name:
            dataset_collection = Permutation(dataset_collection, feature_keys)
            return Diffusion(dataset_collection, max_length)
    else:
        raise NotImplementedError()