import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import numpy as np

'''
        Visual Attention Mechanism with Uniform Frame Sampling

'''


class Frames_SDF_VisualAttention(nn.Module):
    def __init__(self, 
                 
                 image_size,
                 
                 number_of_couples, 
                 
                 frame_patch_size=4, #patch 2D (frame_patch_size,frame_patch_size) da estrarre da ciascun frame
                 
                 sdf_3d_patch_size=4, #patch 3D (sdf_3d_patch_size,sdf_3d_patch_size,sdf_3d_patch_size) da estrarre dal volume sdf
                 
                 embedding_middle_attention = 32, #embedding dentro ogni self attention (quindi feature dimension delle matrici Q,V e K)
                 
                 num_heads=8, #numero di heads per la multihead-attention
                 
                 final_number_of_channels = 4, #alla fine voglio un tensore (_, final_number_of_channels, image_size, image_size, image_size)
                                               #NB: se non vorrai estrarre i patch 3D dell'sdf, allora verrà settato al minimo 2
                 make_sdf_patches = True, #se False non verranno calcolate le patch 3D dell'SDF (quindi l'intera Visual Attention) e questo verrà inserito semplicemente come canale cosi com'è
                 ):
        super().__init__()
        #print("Inizializzo Frames-SDF Visual Attention")

        #64
        self.image_size = image_size
        #4
        self.frame_patch_size = frame_patch_size
        #4 --> (4,4)
        self.number_of_couples = number_of_couples
        #4 --> (4,4,4)
        self.sdf_3d_patch_size = sdf_3d_patch_size
        #se voglio calcolare le patch del volume sdf (e quindi applicargli l'attenzione) oppure includerlo semplicemente cosi com'è come canale (_,1,64,64,64)
        self.make_sdf_patches = make_sdf_patches
        #alla fine voglio un tensore (batch_size, final_number_of_channels, 64, 64, 64)
        if self.make_sdf_patches:
            self.final_number_of_channels = final_number_of_channels if final_number_of_channels>1 else 1
        else:
            self.final_number_of_channels = (final_number_of_channels-1) if final_number_of_channels>=2 else 1

        #embedding delle patch dei frame e dell'input SDF prima dell'attenzione
        self.embedding_before_attention = self.image_size
        #embedding delle matrici Q,K,V
        self.embedding_middle_attention = embedding_middle_attention
        #numero di heads per la multihead attention
        self.num_heads = num_heads

        #Usa il frame_patch di default se non è divisibile per image_size in maniera intera
        if (self.image_size % self.frame_patch_size)!=0:
            print("La size dell'immagine deve essere divisibile per il patch size del frame. Imposto patch frame al valore 4 di default")
            self.frame_patch_size = 4
        
        #Usa l'sdf_patch_size di default se non è divisibile per image_size in maniera intera
        if (self.image_size % self.frame_patch_size)!=0:
            print("La size del volume sdf deve essere divisibile per il patch size 3d scelto. Imposto patch sdf al valore 4 di default")
            self.sdf_3d_patch_size = 4
        

        #creo embedder dei frame prima dell'attenzione
        self.frames_patch_embedder_before_attention = nn.Linear(in_features=(self.frame_patch_size*self.frame_patch_size)*4, out_features=self.embedding_before_attention)

        if self.make_sdf_patches:
            #creo modulo di convoluzione 3D per estrarre le patch dal volume sdf
            self.sdf_patch_extractor = nn.Conv3d(in_channels=1,out_channels=self.embedding_before_attention, kernel_size=(self.sdf_3d_patch_size,self.sdf_3d_patch_size,self.sdf_3d_patch_size), stride=(self.sdf_3d_patch_size,self.sdf_3d_patch_size,self.sdf_3d_patch_size))

        if self.make_sdf_patches:
            #matrice learnable di positional embeddings per le patch dell'sdf e dei frames
            self.positional_embeddings = nn.Parameter(torch.randn(((int(self.image_size/self.sdf_3d_patch_size))**2)*self.number_of_couples + (int(self.image_size/self.sdf_3d_patch_size))**3, self.embedding_before_attention))
        else:
            #matrice learnable di positional embeddings per le patch dei frames
            self.positional_embeddings = nn.Parameter(torch.randn(((int(self.image_size/self.sdf_3d_patch_size))**2)*self.number_of_couples, self.embedding_before_attention))
        
        '''
            Parametri per la Visual Attention
        '''
        self.WQ =  nn.Parameter(torch.randn(self.num_heads, self.embedding_before_attention, self.embedding_middle_attention))
        self.WK =  nn.Parameter(torch.randn(self.num_heads, self.embedding_before_attention, self.embedding_middle_attention))
        self.WV =  nn.Parameter(torch.randn(self.num_heads, self.embedding_before_attention, self.embedding_middle_attention))

        #embedding dopo la multi head attention
        self.W = nn.Parameter(torch.randn(self.num_heads*self.embedding_middle_attention, self.embedding_before_attention))

        if self.make_sdf_patches:
            #layer normalization delle patch dell'sdf e dei frames
            self.layer_normalization = torch.nn.LayerNorm((((int(self.image_size/self.sdf_3d_patch_size))**2)*self.number_of_couples + (int(self.image_size/self.sdf_3d_patch_size))**3,self.embedding_before_attention))
        else:
            #layer normalization delle patch dei frames
            self.layer_normalization = torch.nn.LayerNorm((((int(self.image_size/self.sdf_3d_patch_size))**2)*self.number_of_couples, self.embedding_before_attention))
        
        '''
            Adattamente al ground truth
            Dopo la visual self attention otteniamo una matrice tot_num_patches x 64.
            tot_num_patches è sempre divisibile per 64, quindi se C sono i blocchi in cui le patch
            possono essere divise, faccio il reshape come (batch_size, c, 64, 64).
            Dopo applico una convoluzione 1D in maniera tale da avere (batch_size, 64, 64, 64)
        '''
        if self.make_sdf_patches:
            self.tot_num_patches = ((int(self.image_size/self.sdf_3d_patch_size))**2)*self.number_of_couples + (int(self.image_size/self.sdf_3d_patch_size))**3
        else:
            self.tot_num_patches = ((int(self.image_size/self.sdf_3d_patch_size))**2)*self.number_of_couples
        
        self.num_blocks = int(self.tot_num_patches/self.image_size)

        self.adaptive_conv2d = nn.Conv2d(in_channels=self.num_blocks, out_channels=self.image_size*self.final_number_of_channels,kernel_size=1)
        
    def frames_to_patches(self,frames):
        #print("Converto i frames in patches")

        #tolgo la seconda dimensione in quanto sempre 1
        frames = torch.squeeze(frames, dim=1)

        '''
            Converto ogni canale in patch.
            Ogni patch ha 16 canali, quindi li flatto tutti. Avrò pertanto
            una matrice di lunghezza quanto le patch ma di larghezza quanto le patch flattate dei canali lungo la direzione della patch
        '''
        frames = frames.permute(0,2,3,1)
        patched_frames = Rearrange('b (h i) (w j) c -> b (h w) i j c', i=self.frame_patch_size, j=self.frame_patch_size)(frames)
        patched_frames = patched_frames.permute(0,1,4,2,3)
        chunks = torch.tensor_split(patched_frames, self.number_of_couples, dim=2) 
        patched_frames = torch.cat(chunks, dim=1)
        patched_frames = torch.flatten(patched_frames, start_dim=2)

        #converto gli embedding da 64 a 128
        patched_frames_embedded = self.frames_patch_embedder_before_attention(patched_frames)

        #(batch_size, 1024, 128)
        return patched_frames_embedded

    def sdf_to_patches(self, sdf):
        #print("Converto SDF in patches")

        '''
            La convoluzione crea 64 features, ossia 64 volumi.
            Ogni singolo volume ha dimensione 61x61x61. Ogni patch
            2D del canale del volume indica una patch. Pertanto ogni
            patch ha 64 feature distribuite nelle stesse posizioni ma
            dei 64 volumi.
        '''
        #(batch_size, 64, 16, 16, 16)
        patched_sdf = self.sdf_patch_extractor(sdf)

        batch_size, channels, H, W, L = patched_sdf.shape

        #ogni valore di ogni volume 61x61x61 ha la successiva feature nella stessa posizione ma nel volume successivo.
        #raggrupo quindi le features sui 64 volumi
        #(batch_size, C, H*W*L)
        patched_sdf_view1 = patched_sdf.view(batch_size, channels, -1)
        #(batch_size, H*W*L, C)
        patched_sdf_embeddings = patched_sdf_view1.permute(0,2,1)

        #(batch_size, 4096, 64)
        return patched_sdf_embeddings


    def add_positional_encoding_to_embeddings(self, embeddings):
        
        return torch.add(embeddings, self.positional_embeddings)

    def multi_head_attention(self, condition_embeddings_with_pos):

        X_concatenated = None
        #per il numero di head scelto...
        for head_i in np.arange(self.num_heads):

            X_new = self.self_attention(condition_embeddings_with_pos, head_i)
            X_concatenated = torch.cat((X_concatenated, X_new), dim=2) if X_concatenated!=None else X_new

        #trasformazione finale per ridurre le dimensioni di concatenazione
        X_new = torch.matmul(X_concatenated, self.W)

        #skip layer (sommo X originale)
        X_new_with_originals = X_new + condition_embeddings_with_pos

        #normalizzo
        X_new_with_originals_normalized = self.layer_normalization(X_new_with_originals)

        return X_new_with_originals_normalized
        

    def self_attention(self, X, head):

        #calcolo prima trasformazione lineare 
        Q = torch.matmul(X,self.WQ[head])
        #calcolo seconda trasformazione lineare
        K = torch.matmul(X,self.WK[head])
        #calcolo terza trasformazione lineare
        V = torch.matmul(X,self.WV[head])

        #dot product
        dot_product_attention = torch.bmm(Q,torch.transpose(K,1,2))

        #scaling (necessario per evitare che la softmax abbia piccoli gradienti)
        scaled_dot_product_attention = dot_product_attention / torch.sqrt(torch.tensor(self.embedding_middle_attention, dtype=torch.float32))

        #trasformo in percentuali le somiglianze scalate
        attention_filter = torch.softmax(scaled_dot_product_attention, dim=2)

        #applico l'ultima trasformazione
        new_X = torch.bmm(attention_filter, V)

        return new_X

    def adapt(self, X):
        
        batch_size, num_patches, num_features = X.shape
        
        #riarrangio il tensore in (batch_size, C, 64, 64)
        X_reshaped = X.view((batch_size, self.num_blocks, self.image_size, self.image_size))
        
        #applico convoluzione 1D per ottenere (batch_size, 64, 64, 64)
        X_convoluted = self.adaptive_conv2d(X_reshaped)
        X_convoluted = X_convoluted.reshape(batch_size, self.final_number_of_channels, self.image_size, self.image_size, self.image_size)

        return X_convoluted


    def forward(self, cond1, cond2):

        #prendo l'input SDF (1, 64, 64, 64)
        input_sdf = cond1

        #prendo i frames (1, 16, 64, 64)
        frames = cond2

        '''
            Creo patch dei frames. Ogni riga corrisponderà
            a un patch nel canale R, G, B e classe del frame.
            Avremo quindi 1024 patch ciascuna con embedding 64.

            Più in giu vai più spazialmente al frame ti muovi. Ovviamente,
            essendo le patch una sotto l'altra anche degli altri frame, più
            sotto vai più "temporalmente" ti stai spostando.
        '''
        #(batch_size, 1024, 64)
        patched_frames_embeddings = self.frames_to_patches(frames)

        '''
            Creo patch 3D dell'input SDF.
        '''
        if self.make_sdf_patches:
            #(batch_size, 4096, 64)
            patched_sdf_embeddings = self.sdf_to_patches(input_sdf)
            #metto uno sotto l'altro gli embeddings (batch_size, 4096+1024, 64)
            condition_embeddings = torch.cat((patched_sdf_embeddings, patched_frames_embeddings), dim=1)
        else:
            condition_embeddings = patched_frames_embeddings

        #li nutro dell'informazione sulla posizione
        condition_embeddings_with_pos = self.add_positional_encoding_to_embeddings(condition_embeddings)

        #applica la multihead-attention
        condition_embeddings_with_pos_after_visual_attention = self.multi_head_attention(condition_embeddings_with_pos)

        #adatto la condition alla forma del ground truth
        #(batch_size, final_number_of_channels [-1 se make_sdf_patches==False], 64, 64, 64)
        condition = self.adapt(condition_embeddings_with_pos_after_visual_attention)

        #se ho eseguito patches solo sui frame, aggiungo l'sdf come semplice altro canale
        if self.make_sdf_patches==False:
            #(1, final_number_of_channels, 64, 64, 64)
            return torch.cat((condition, input_sdf), dim=1)
        
        return condition


if __name__ == '__main__':
    
    cond1 = torch.randn((2,1,32,32,32), dtype=torch.float32)
    cond2 = torch.randn((2,1,24,32,32), dtype=torch.float32)

    model = Frames_SDF_VisualAttention(image_size=32,number_of_couples=6)

    print(model(cond1,cond2).shape)