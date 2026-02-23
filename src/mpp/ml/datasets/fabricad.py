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
from mpp.constants import PATHS, VOCAB, INV_VOCAB

class Fabricad(Dataset):
    """
    PyTorch Dataset for loading synthetic or real manufacturing process data,
    including structured part features and process plan targets.

    The dataset supports multiple input/output formats, including:
    - Input: structured feature vectors ("vecset")
    - Target: total time, total cost, ordered process sequence, or multilabel step set

    Parameters
    ----------
    mode : str, optional
        Dataset split to use. One of {'train', 'valid', 'test'}. Default is 'train'.
    transform : callable, optional
        Optional transform to be applied on the input sample.
    target_transform : callable, optional
        Optional transform to be applied on the target.
    target_type : str, optional
        Type of prediction target. One of {'time', 'cost', 'step-set', 'seq'}.
    input_type : str, optional
        Type of input features. Currently only 'vecset' is supported.

    Raises
    ------
    ValueError
        If unsupported input_type or target_type is provided.
    """

    def __init__(self, mode = "train", transform=None, target_transform=None, target_type = "seq", input_type="vecset"):
        self.cache = {}

        # class vars
        self.target_type = target_type
        self.input_type = input_type

        assert self.target_type in ["time", "cost", "step-set", "seq"], ValueError(f"Not supported prediction type: {self.target_type}")
        assert self.input_type in ["vecset"], ValueError(f"Not supported input type: {self.input_type}")

        # check if a split file exists, if not create one
        split_file = PATHS.CONFIG/ "sample_split.json"

        if not split_file.exists():
            self.split()

        with open(split_file, "r") as f:
            split_dict = json.load(f)

        # load the samples from the split file
        self.samples = split_dict[mode]

        #self.plan_dir = PATHS.SYNTHETIC_PP_DATA
        logger.info(f"Dataset initialized with {len(self.samples)} samples for {mode} mode with input_type: {self.input_type.upper()} target_type: {self.target_type.upper()}.")


    def split(self, train_size=0.8, valid_size=0.1, test_size=0.1):
        """
        Randomly splits available samples into train, validation, and test subsets.

        Parameters
        ----------
        train_size : float
            Proportion of samples for the training set.
        valid_size : float
            Proportion of samples for the validation set.
        test_size : float
            Proportion of samples for the test set.

        Raises
        ------
        ValueError
            If the proportions do not sum to 1.0.
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
        with open(PATHS.CONFIG/ "sample_split.json", "w") as f:
            json.dump(split_dict, f, indent=4)

        logger.info(f"Split dataset into {len(train_samples)} train, {len(valid_samples)} validation and {len(test_samples)} test samples.")

        return None
    

    def parse_part(self, idx):
        """
        Loads and returns one data sample: input features and corresponding target.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple(torch.Tensor, torch.Tensor or float)
            Input vector and corresponding target value, depending on target_type.
        """

        # parse item data from files
        input_item = self.parse_input_item(idx)
        target_item = self.parse_target_item(idx)

        logger.debug(f"get the items: Input: {input_item.shape} for type {self.input_type}, Target: {target_item} for type: {self.target_type}")

        return input_item, target_item

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns the indexed sample. Uses an internal cache to avoid reloading samples.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        tuple(torch.Tensor, torch.Tensor or float)
            Input and target of the sample.
        """

        input_item = None
        target_item = None

        if idx in self.cache.keys():
            input_item, target_item = self.cache[idx]
        else:
            input_item, target_item = self.parse_part(idx)
            self.cache[idx] = (input_item, target_item)
        
        return input_item, target_item
    
    #utils
    @staticmethod
    def encode_sequence(seq):
        """
        Encodes a list of process step tokens into their corresponding indices using VOCAB.

        Parameters
        ----------
        seq : list of str
            Sequence of process steps (e.g., ["START", "Bohren", "STOP"]).

        Returns
        -------
        list of int
            List of encoded token indices.
        """
        return [VOCAB[token] for token in seq]
    
    @staticmethod
    def decode_sequence(seq):
        """
        Decodes a list of token indices into their corresponding step names using INV_VOCAB.

        Parameters
        ----------
        seq : list[int] or torch.Tensor
            Encoded sequence of token indices.

        Returns
        -------
        list of str
            Decoded sequence of process steps.
        """
        return [INV_VOCAB[int(token)] for token in seq]
    
    def get_multilabel_targets(self, steps : list[str]):
        """
        Creates a multi-label one-hot encoded target vector from a list of process steps.

        Parameters
        ----------
        steps : list of str
            List of process steps present in the current sample.

        Returns
        -------
        torch.Tensor
            1D binary tensor of shape [num_classes] indicating which steps are present.
        """
        all_possible_steps = [elem for elem in list(VOCAB.keys()) if elem not in ["START", "STOP", "PAD"]] #extract all standard processes defined in constants
        num_classes = len(all_possible_steps) #set the dimension of the target vector

        targets = torch.zeros(num_classes) # init target vector with zeros
        idxs = [VOCAB[step] for step in set(steps)] #get indexes of processes to set them in the targets vector

        targets[idxs] = 1 # set the classes 

        return targets  
    
    def parse_target_item(self, idx):
        """
        Parses the target item (label) for a given sample based on the specified target_type.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        float or torch.Tensor
            The computed target value:
            - np. float64 (for time or cost)
            - Tensor (for step-set or sequence)
        """
        # load plan
        plan_item = pd.read_csv(PATHS.FEATURE_DATA / self.samples[idx] / 'plan.csv', sep=';')
        steps = plan_item["Schritt"].tolist()[1:]
        steps.remove("liefern")

        #calculate target item
        match self.target_type:
            case "time":
                return torch.tensor(plan_item["Dauer[min]"].sum()).float()
            case "cost":
                return plan_item["Kosten[($)]"].sum()
            case "step-set":
                return self.get_multilabel_targets(steps) # multiclass target vector for binary or multiclass classification
            case "seq":
                wrapped_steps = ["START"] + steps + ["STOP"]
                return torch.Tensor(self.encode_sequence(wrapped_steps))
            
            # TODO here additional items can be added
        return None
    
    def parse_input_item(self, idx):
        """
        Loads the input vector (e.g., vecset) for a given sample.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        torch.Tensor
            Input vector tensor.
        """
        match self.input_type:
            case "vecset":
                vecset_item = PATHS.FEATURE_DATA / self.samples[idx] / "features/vecset.npy"
                return torch.Tensor(np.load(vecset_item))
        




# validate if the dataset is working
if __name__ == "__main__":
    for input_type in ["vecset"]:
        for target_type in ["time", "cost", "step-set", "seq"]:
            dataset = Fabricad(input_type=input_type, target_type=target_type)
            input_item, target_item = dataset[11]
            logger.info(f"get the items: Input: {input_item.shape} for type {input_type}, Target: {target_item} for type: {target_type}")
