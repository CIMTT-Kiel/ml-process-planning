import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import json
import logging

# set up logger

logging.basicConfig(
    format="%(asctime)s %(levelname)8s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s %(levelname)8s - %(message)s")


#custom imports
from cadtoseq.constants import PATHS, VOCAB, INV_VOCAB


def encode_sequence(seq):
    return [VOCAB[token] for token in seq]

class Fabricad(Dataset):
    def __init__(self, mode = "train", transform=None, target_transform=None):
        self.cache = {}

        # check if a split file exists, if not create one
        split_file = PATHS.FEATURE_DATA / "sample_split.json"

        if not split_file.exists():
            self.split()

        with open(split_file, "r") as f:
            split_dict = json.load(f)

        # load the samples from the split file
        self.samples = split_dict[mode]

        #self.plan_dir = PATHS.SYNTHETIC_PP_DATA
        logger.info(f"Dataset initialized with {len(self.samples)} samples for {mode} mode.")


        # optional transformations
        self.transform = transform
        self.target_transform = target_transform

    def split(self, train_size=0.8, valid_size=0.1, test_size=0.1):
        """split splits the dataset into train, validation and test sets
        """
        logger.info("Splitting dataset into train, validation and test sets...")
        if train_size + valid_size + test_size != 1.0:
            raise ValueError("train_size, valid_size and test_size must sum to 1.0")

        all_samples = [path.stem for path in PATHS.FEATURE_DATA.iterdir() if path.stem!= ".DS_Store" and path.is_dir()]
        
        # shuffle the samples
        np.random.shuffle(all_samples)
        n_samples = len(all_samples)

        train_end = int(train_size * n_samples)
        valid_end = int((train_size + valid_size) * n_samples)

        train_samples = all_samples[:train_end]
        valid_samples = all_samples[train_end:valid_end]
        test_samples = all_samples[valid_end:]

        #create split dictionary
        split_dict = {
            "train": train_samples,
            "valid": valid_samples,
            "test": test_samples
        }
        #save split dictionary to json file
        with open(PATHS.FEATURE_DATA / "sample_split.json", "w") as f:
            json.dump(split_dict, f, indent=4)

        logger.info(f"Split dataset into {len(train_samples)} train, {len(valid_samples)} validation and {len(test_samples)} test samples.")

        return None
    

    def parse_part(self, idx):
        """parse_part loads the vecset and plan item
        """
        vecset_item = PATHS.FEATURE_DATA / self.samples[idx] / "features/vecset.npy"
        plan_item = PATHS.FEATURE_DATA / self.samples[idx] / 'production_plan/production_plan.json'


        # load sample
        vecset_item = torch.Tensor(np.load(vecset_item))


        # load plan
        plan_item = pd.read_json(plan_item)
        steps = plan_item["Schritt"].tolist()[1:] + ["STOP"]


        plan_item = torch.Tensor(encode_sequence(steps))


        # apply transformations
        if self.transform:
            vecset_item = self.transform(vecset_item)
        if self.target_transform:
            plan_item = self.target_transform(plan_item)

        return vecset_item, plan_item

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """__getitem__ checks if image file is in chache -> self.data. If not the function parse_part is called to apply preprocessing.
        if only 2D or 3D is needed the function will return an empty Tensor for the other item, but al
        """

        vecset_item = None
        plan_item = None

        if idx in self.cache.keys():
            vecset_item, plan_item = self.cache[idx]
        else:
            vecset_item, plan_item = self.parse_part(idx)
            self.cache[idx] = (vecset_item, plan_item)
        
        return vecset_item, plan_item
    
    #utils
    def encode_sequence(seq):
        return [VOCAB[token] for token in seq]
    
    @staticmethod
    def decode_sequence(seq):
        
        return [INV_VOCAB[int(token)] for token in seq]


# validate if the dataset is working
if __name__ == "__main__":
    dataset = Fabricad()
    vecset_item, plan_item = dataset[11]
    logging.info(vecset_item.shape)
    logging.info(plan_item)