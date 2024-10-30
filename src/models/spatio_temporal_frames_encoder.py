import torch.nn as nn
import torch.nn.functional as F
import torch

'''
    Converte i frames (b,1,24,64,64) in un tensore (b,1,64,64,64)

    NB: VALIDO SOLO PER 6 COPPIE!

'''



'''
                RESIDUAL BLOCK PER I FRAMES  

                Un residual block in genere non altera il numero di canali in ingresso, ma qui ne permetteremo la possibilità
'''

class Residual_Block(nn.Module):
    def __init__(self,
                 in_channels, #canali in ingresso
                 out_channels, #canali in uscita
                 groups=4, #per le statistiche divide il tensore prendendo groups canali
                 kernel_size = 3, #kernel size della convoluzione
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.groups = groups
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        #PRIMA NORMALIZZAZIONE E CONVOLUZIONE
        self.groupnormLayer1 = nn.GroupNorm(groups, in_channels)
        self.conv2D_1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding="same")

        #SECONDA NORMALIZZAZIONE E CONVOLUZIONE
        self.groupnormLayer2 = nn.GroupNorm(groups, out_channels)
        self.conv2D_2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding="same")

        #se i canali di ingresso e quelli di uscita sono diversi dovrò fare un'altra convoluzione, ma stavolta su x originale
        #per portarlo come quelli di uscita dato che poi l'ultimo tensore verrà sommato al tensore originale
        if in_channels == out_channels:
            self.residualLayer = nn.Identity()
        else:
            self.residualLayer = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):

        y = self.groupnormLayer1(x)
        y = F.silu(y)
        y = self.conv2D_1(y)

        y= self.groupnormLayer2(y)
        y = F.silu(y)
        y = self.conv2D_2(y)

        return y + self.residualLayer(x)

class SpatioTemporalHierarchicalFeatures(nn.Module):
    def __init__(self, number_of_couples , image_size):
        super().__init__()

        self.number_of_couples = number_of_couples
        self.image_size = image_size

        #Prima convoluzione -->(b, 1, 6*4, 64, 64) --> (b, 3, 6, 32, 32)
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=10, kernel_size=(4,4,4), stride=(4,2,2), padding=(0,1,1))
        self.residualblock1 = Residual_Block(in_channels=10, out_channels=6, groups=2) 
        self.residualblock12 = Residual_Block(in_channels=6, out_channels=2, groups=1)
        #--> (b, 2, 6, 64, 64)
        self.upsample1 = nn.Upsample(size=(6,64,64))  

        
        #Seconda convoluzione -->(b, 1, 6*4, 64, 64) --> (b, 2, 5, 32, 32)
        self.conv2 = nn.Conv3d(in_channels=1, out_channels=10, kernel_size=(8,4,4), stride=(4,2,2), padding=(0,1,1))
        self.residualblock2 = Residual_Block(in_channels=10, out_channels=6, groups=2) 
        self.residualblock22 = Residual_Block(in_channels=6, out_channels=2, groups=1) 
        #--> (b, 2, 5, 64, 64)
        #upsampling
        self.upsample2 = nn.Upsample(size=(5,64,64)) 
        
        #Terza convoluzione -->(b, 1, 6*4, 64, 64) --> (b, 3, 4, 32, 32)
        self.conv3 = nn.Conv3d(in_channels=1, out_channels=10, kernel_size=(12,4,4), stride=(4,2,2), padding=(0,1,1))
        self.residualblock3 = Residual_Block(in_channels=10, out_channels=6, groups=2) 
        self.residualblock32 = Residual_Block(in_channels=6, out_channels=3, groups=1) 
        #--> (b, 3, 4, 64, 64)
        #upsampling
        self.upsample3 = nn.Upsample(size=(4,64,64)) 

        #Quarta convoluzione -->(b, 1, 6*4, 64, 64) --> (b, 4, 3, 32, 32)
        self.conv4 = nn.Conv3d(in_channels=1, out_channels=10, kernel_size=(16,4,4), stride=(4,2,2), padding=(0,1,1))
        self.residualblock4 = Residual_Block(in_channels=10, out_channels=8, groups=2) 
        self.residualblock42 = Residual_Block(in_channels=8, out_channels=4, groups=1) 
        #--> (b, 4, 3, 64, 64)
        #upsampling
        self.upsample4 = nn.Upsample(size=(3,64,64)) 

        #Quarta convoluzione -->(b, 1, 6*4, 64, 64) --> (b, 5, 2, 32, 32)
        self.conv5 = nn.Conv3d(in_channels=1, out_channels=10, kernel_size=(20,4,4), stride=(4,2,2), padding=(0,1,1))
        self.residualblock5 = Residual_Block(in_channels=10, out_channels=10, groups=2) 
        self.residualblock52 = Residual_Block(in_channels=10, out_channels=5, groups=1) 
        #--> (b, 5, 2, 64, 64)
        #upsampling
        self.upsample5 = nn.Upsample(size=(2,64,64)) 

        #Quarta convoluzione -->(b, 1, 6*4, 64, 64) --> (b, 8, 1, 32, 32)
        self.conv6 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(24,4,4), stride=(4,2,2), padding=(0,1,1))
        self.residualblock6 = Residual_Block(in_channels=16, out_channels=16, groups=2) 
        self.residualblock62 = Residual_Block(in_channels=16, out_channels=8, groups=1) 
        #--> (b, 8, 1, 64, 64)
        #upsampling
        self.upsample6 = nn.Upsample(size=(1,64,64)) 

    
    #(b, 1, 24, 64, 64)
    def forward(self, frames):

        #(b, 1, 24, 64, 64) -> (b, 2, 6, 32, 32)
        f1 = self.conv1(frames)
        f1 = self.residualblock1(f1)
        f1 = self.residualblock12(f1)
        #--> (b, 2, 6, 64, 64)
        f1 = self.upsample1(f1)
        b,c,d,w,l = f1.shape
        f1 = f1.permute((0,2,1,3,4))
        #--> (b, 12, 64, 64)
        f1 = f1.reshape((b,c*d,w,l))

        #(b, 1, 24, 64, 64) -> (b, 2, 5, 32, 32)
        f2 = self.conv2(frames)
        f2 = self.residualblock2(f2)
        f2 = self.residualblock22(f2)
        #--> (b, 6, 3, 64, 64)
        f2 = self.upsample2(f2)
        b,c,d,w,l = f2.shape
        f2 = f2.permute((0,2,1,3,4))
        #--> (b, 10, 64, 64)
        f2 = f2.reshape((b,c*d,w,l))

        #(b, 1, 24, 64, 64) -> (b, 3, 4, 32, 32)
        f3 = self.conv3(frames)
        f3 = self.residualblock3(f3)
        f3 = self.residualblock32(f3)
        #--> (b, 8, 2, 64, 64)
        f3 = self.upsample3(f3)
        b,c,d,w,l = f3.shape
        f3 = f3.permute((0,2,1,3,4))
        #--> (b, 12, 64, 64)
        f3 = f3.reshape((b,c*d,w,l))

        #(b, 1, 24, 64, 64) -> (b, 4, 3, 32, 32)
        f4 = self.conv4(frames)
        f4 = self.residualblock4(f4)
        f4 = self.residualblock42(f4)
        #--> (b, 12, 1, 64, 64)
        f4 = self.upsample4(f4)
        b,c,d,w,l = f4.shape
        f4 = f4.permute((0,2,1,3,4))
        #--> (b, 12, 64, 64)
        f4 = f4.reshape((b,c*d,w,l))

        #(b, 1, 24, 64, 64) -> (b, 5, 2, 32, 32)
        f5 = self.conv5(frames)
        f5 = self.residualblock5(f5)
        f5 = self.residualblock52(f5)
        #--> (b, 12, 1, 64, 64)
        f5 = self.upsample5(f5)
        b,c,d,w,l = f5.shape
        f5 = f5.permute((0,2,1,3,4))
        #--> (b, 10, 64, 64)
        f5 = f5.reshape((b,c*d,w,l))

        #(b, 1, 24, 64, 64) -> (b, 8, 1, 32, 32)
        f6 = self.conv6(frames)
        f6 = self.residualblock6(f6)
        f6 = self.residualblock62(f6)
        #--> (b, 12, 1, 64, 64)
        f6 = self.upsample6(f6)
        b,c,d,w,l = f6.shape
        f6 = f6.permute((0,2,1,3,4))
        #--> (b, 8, 64, 64)
        f6 = f6.reshape((b,c*d,w,l))
        
        #(b, 64,64,64)
        condition2 = torch.unsqueeze(torch.cat((f1,f2,f3,f4,f5,f6), dim=1), dim=1)

        return condition2




# if __name__ == '__main__':

#     number_of_couples = 6
#     image_size = 64

#     condition2 = torch.randn((2, 1, 6*4, image_size, image_size), dtype=torch.float32)

#     model = SpatioTemporalHierarchicalFeatures(number_of_couples, image_size)

#     model(condition2)