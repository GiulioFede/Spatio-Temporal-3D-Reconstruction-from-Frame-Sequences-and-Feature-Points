import torch
import torch.nn as nn
import numpy as np
from transformers import VivitImageProcessor, VivitModel, VivitConfig


class Frames_SDF_Fusion(nn.Module):
    def __init__(
        self,
        number_of_couples
    ):
        super().__init__()

        self.number_of_couples = number_of_couples

        self.temporal_convolution = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(8,16,16), stride=(8,16,16))

        self.augment_frame = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=1, stride=1)

        self.pixel_weights = nn.Parameter(torch.zeros(64,64,64), dtype=torch.float32)


    def forward(self, frames):

        
        #(b, 1, 16, 4, 64, 64)
        frames = frames.view(frames.shape[0],1, self.number_of_couples, 4, 64, 64)

        #(b, 1, 16, 64, 64, 64)
        frame = self.augment_frame(frame)

        #
        frame = nn.Tanh()(frame)



        #sdf = sdf + frame_1*p_1 + frame_2*p2 + frame_3*p3
            


        return 


if __name__ == '__main__':

    image_size = 64
    number_of_couples = 32
    
    #cond1 = torch.randn((2,1,32,32,32), dtype=torch.float32)
    sdf = torch.randn((2,1,image_size,image_size,image_size), dtype=torch.float32)
    frames = torch.randint(0, 255, (2,1,number_of_couples*4,image_size,image_size), dtype=torch.float32)

    model = Frames_SDF_Fusion(number_of_couples)

    print(model(frames).shape)