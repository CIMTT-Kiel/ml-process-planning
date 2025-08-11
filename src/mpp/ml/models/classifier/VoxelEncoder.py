import torch.nn as nn
import torch
import numpy as np

class VoxelEncoder(nn.Module):
    def __init__(self, base_channel_size=4, latent_size=2, act_fn=nn.LeakyReLU(), pool=nn.MaxPool3d(kernel_size=3)):
        super().__init__()
        self.c_hid = base_channel_size
        
        self.mlp_input=8*8*8*16
        self.latent_size=latent_size
  
        self.cnn_encoder = nn.Sequential(
            nn.Conv3d(1, 2*self.c_hid, kernel_size=5), 
            act_fn,
            pool,
            nn.Conv3d(2 * self.c_hid, 3 * self.c_hid, kernel_size=5),
            act_fn,
            pool,
            nn.Conv3d(3 * self.c_hid, 4 * self.c_hid, kernel_size=5), 
            act_fn
        )
        
        self.mlp_head = nn.Sequential(
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(self.mlp_input, int(self.mlp_input*(2/3))),
            act_fn,
            nn.Linear(int(self.mlp_input*(2/3)), self.latent_size),
            nn.Sigmoid(),
            )


    def encode(self, x):
        x=self.cnn_encoder(x)
        x=self.mlp_head(x)


        return x  

class VoxelEncoderProductionStep(nn.Module):
    def __init__(self, base_channel_size=4, latent_size=5, act_fn=nn.LeakyReLU(), pool=nn.MaxPool3d(kernel_size=3)):
        super().__init__()
        self.c_hid = base_channel_size
        
        self.mlp_input=8*8*8*16
        self.latent_size=latent_size
  
        self.cnn_encoder = nn.Sequential(
            nn.Conv3d(1, 2*self.c_hid, kernel_size=5), 
            act_fn,
            pool,
            nn.Conv3d(2 * self.c_hid, 3 * self.c_hid, kernel_size=5),
            act_fn,
            pool,
            nn.Conv3d(3 * self.c_hid, 4 * self.c_hid, kernel_size=5), 
            act_fn
            # pool,
            # nn.Conv3d(4 * self.c_hid, 4 * self.c_hid, kernel_size=3),
            # act_fn,
            # nn.Conv3d(4 * self.c_hid, 4 * self.c_hid, kernel_size=5),  
            # act_fn,
            # pool
        )
        
        self.mlp_head = nn.Sequential(
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(self.mlp_input, int(self.mlp_input*(2/3))),
            act_fn,
            nn.Linear(int(self.mlp_input*(2/3)), self.latent_size),
            act_fn,
            )


    def encode(self, x):
        x=self.cnn_encoder(x)
        x=self.mlp_head(x)


        return x  

#testsection

if __name__ == "__main__":
    voxel=torch.Tensor(np.random.rand(1,1,128,128,128))
    encoder=VoxelEncoder()
    latent=encoder.encode(voxel)
