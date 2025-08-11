#standard library imports
import logging

#third party imports
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

#custom imports
from cadtoseq.constants import VOCAB
from cadtoseq.ml.datasets.fabricad import Fabricad

logging.basicConfig(
    format="%(asctime)s %(levelname)8s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s %(levelname)8s - %(message)s")

def collate_fn(batch):
    vecsets, plans = zip(*batch)
    vecsets = torch.stack(vecsets)

    max_len = 10
    padded_plans = torch.full((len(plans), max_len), VOCAB["PAD"], dtype=torch.long)

    for i, plan in enumerate(plans):
        padded_plans[i, :plan.size(0)] = plan

    return vecsets, padded_plans

class Fabricad_datamodule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4):
        super().__init__()
        logger.info("Initializing Fabricad datamodule")
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Initialize datasets for actual stage
        logger.info(f"Setting up Fabricad datamodule for stage: {stage}")
        if stage == "fit" or stage is None:
            self.train_dataset = Fabricad(mode="train")
            self.val_dataset = Fabricad(mode="valid")
            logger.info(f"Train dataset size: {len(self.train_dataset)}, Validation dataset size: {len(self.val_dataset)}")

        if stage == "test" or stage is None:
            self.test_dataset = Fabricad(mode="test")
            logger.info(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self):
        logger.debug("Creating train dataloader")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)

    def val_dataloader(self):
        logger.debug("Creating validation dataloader")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)

    def test_dataloader(self):
        logger.debug("Creating test dataloader")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)


#checks
if __name__ == "__main__":
    # Example usage
    vecset_data_module = Fabricad_datamodule(batch_size=32, num_workers=4)
    vecset_data_module.setup(stage="fit")
    
    train_loader = vecset_data_module.train_dataloader()
    validation_loader = vecset_data_module.val_dataloader()

    train_batch = next(iter(train_loader))
    validation_batch = next(iter(validation_loader))

    print("Train batch shape:", train_batch[0].shape, train_batch[1].shape)
    print("Validation batch shape:", validation_batch[0].shape, validation_batch[1].shape)

    print("Sequences:", train_batch[1])
