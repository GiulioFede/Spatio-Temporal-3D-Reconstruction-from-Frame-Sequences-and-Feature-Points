import torch
import torch.nn as nn
import numpy as np


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Args:
        x: Tensor representing the image of shape [B, C, H, W]
        patch_size: Number of pixels per dimension of the patches (integer)
        flatten_channels: If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """Attention Block.

        Args:
            embed_dim: Dimensionality of input and attention feature vectors
            hidden_dim: Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads: Number of heads to use in the Multi-Head Attention block
            dropout: Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        patch_size,
        dropout=0.0,
        image_size=128,
        number_of_couples=30, 
        final_dimension=16,
        what_to_return = "sdf"
    ):
        """Vision Transformer.

        Args:
            embed_dim: Dimensionality of the input feature vectors to the Transformer
            hidden_dim: Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels: Number of channels of the input (3 for RGB)
            num_heads: Number of heads to use in the Multi-Head Attention block
            num_layers: Number of layers to use in the Transformer
            patch_size: Number of pixels that the patches have per dimension
            num_patches: Maximum number of patches an image can have
            dropout: Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size
        self.number_of_couples = number_of_couples
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.what_to_return = what_to_return

        self.num_patches = (int(image_size/patch_size)**2)*number_of_couples

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )

        self.sdf_conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.sdf_conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.sdf_conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.sdf_conv4 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.sdf_conv5 = nn.Conv3d(in_channels=256, out_channels=embed_dim, kernel_size=4, stride=2, padding=1)
        self.sdf_avg_pool = nn.AvgPool3d(kernel_size=2)
        self.sdf_normalization_layer = nn.LayerNorm(normalized_shape=(1, embed_dim))
      
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + self.num_patches, embed_dim))

        if self.what_to_return == "sdf" or what_to_return == "frames":
            self.output_layer_normalization = nn.LayerNorm(normalized_shape=(1,64,64,64))

        if self.what_to_return == "sdf" or self.what_to_return=="both":
            self.encode_final_sdf = nn.Linear(in_features=self.embed_dim, out_features=64*64*64)
        
        if self.what_to_return == "both":
            self.output_layer_normalization_frames = nn.LayerNorm(normalized_shape=(1,64,64,64))
            self.output_layer_normalization_sdf = nn.LayerNorm(normalized_shape=(1,64,64,64))

    def encode_sdf(self, sdf):

        sdf = self.sdf_conv1(sdf)
        sdf = self.sdf_conv2(sdf)
        sdf = self.sdf_conv3(sdf)
        sdf = self.sdf_conv4(sdf)
        sdf = self.sdf_conv5(sdf)
        sdf = self.sdf_avg_pool(sdf)
        #(b, 1, 512)
        sdf = sdf.squeeze(-1).squeeze(-1).squeeze(-1).unsqueeze(1)
        sdf = self.sdf_normalization_layer(sdf)
        sdf = nn.Tanh()(sdf)

        return sdf

    def forward(self, x, sdf):

        #(b, n_couples, 128, 128)
        frames = x.squeeze(1)
        x = None
        # Preprocess input
        for couple_i in np.arange(0, self.number_of_couples*4, step=4):
            frame_i = frames[:,couple_i:couple_i+4]
            #(b, num_patches_per_image, embedding)
            #NB: ogni patch è una versione flatten del canale rosso, verde, blue e classe. Quindi ognuno ha lunghezza 16x16 e quindi in totale 16x16x4
            x = img_to_patch(frame_i, self.patch_size) if couple_i==0 else torch.cat([x, img_to_patch(frame_i, self.patch_size)], dim=1)

        B, T, _ = x.shape
        #riduco l'embedding di ogni patch a 256
        x = self.input_layer(x)

        # da 1024 passo ancora a 1024
        sdf_embedding = self.encode_sdf(sdf)

        x = torch.cat([sdf_embedding, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        sdf = x[0]
        frames = x[1:].permute(1,0,2)

        if self.what_to_return == "sdf":
            sdf = self.encode_final_sdf(sdf)
            sdf = sdf.view(frames.shape[0], 1, 64, 64, 64)
            sdf = self.output_layer_normalization(sdf)
            sdf = nn.Tanh()(sdf)
            return sdf
        elif self.what_to_return == "frames":
            #(2, 512, 16, 8, 8)
            frames = frames.view(frames.shape[0], 1, 64, 64, 64)
            frames = self.output_layer_normalization(frames)
            frames = nn.Tanh()(frames)
            return frames
        else:
            sdf = self.encode_final_sdf(sdf)
            sdf = sdf.view(frames.shape[0], 1, 64, 64, 64)
            sdf = self.output_layer_normalization_sdf(sdf)
            sdf = nn.Tanh()(sdf)

            frames = frames.view(frames.shape[0], 1, 64, 64, 64)
            frames = self.output_layer_normalization_frames(frames)
            frames = nn.Tanh()(frames)

            return torch.cat([frames, sdf], dim=1)




if __name__ == '__main__':

    image_size = 64
    number_of_couples = 16
    
    #cond1 = torch.randn((2,1,32,32,32), dtype=torch.float32)
    sdf = torch.randn((2,1,image_size,image_size,image_size), dtype=torch.float32)
    frames = torch.randn((2,1,number_of_couples*4,image_size,image_size), dtype=torch.float32)

    model = VisionTransformer(embed_dim=16*16*4, 
                              hidden_dim=16*16*4, 
                              num_channels=4, 
                              num_heads=8, 
                              num_layers=6, 
                              patch_size=16, 
                              image_size=image_size,
                              number_of_couples=number_of_couples, 
                              final_dimension=64,
                              what_to_return = "sdf")

    print(model(frames,sdf).shape)