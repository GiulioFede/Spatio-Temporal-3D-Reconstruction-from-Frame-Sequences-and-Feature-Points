import torch
import torch.nn as nn
import numpy as np
from transformers import VivitImageProcessor, VivitModel, VivitConfig


class FRAMES_VisionTransformer(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

        self.number_of_couples = 32


        self.image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2")
        self.model = VivitModel.from_pretrained("google/vivit-b-16x2")
        self.model.eval()
        self.model.config.add_pooling_layer = False


    def reverse_patching(self, emb):

        x = torch.zeros((144,126,126), dtype=torch.float32).cuda()

        c = 0
        for img_i in np.arange(0, emb.shape[0], 196):

            #(196, 729)
            embeddings = emb[img_i:img_i+196, :]

            #splitto i patch ogni 14. Quindi ogni split contiene tutti i patch di ogni riga
            patches_per_row = embeddings.split(14)

            shift_r = 0
            for row in np.arange(len(patches_per_row)):

                shift_c = 0
                for patch_i in np.arange(len(patches_per_row[row])):
                    
                    patch = patches_per_row[row][patch_i].view(9,9,9)

                    x[c:c+9,shift_r:shift_r+9, shift_c:shift_c+9] = patch

                    shift_c += 9
                
                shift_r += 9
        
            c = c + 9

        return x


    def forward(self, frames):

        frames = torch.stack([frames[:,:,c:c+3,:,:].permute(0,1,3,4,2).squeeze(1) for c in np.arange(0, self.number_of_couples*4, step=4)], axis=1)

        output = None

        for b in np.arange(frames.shape[0]):
            preprocess_frames = self.image_processor(list(frames[b,:]), return_tensors="pt")
            #print("prediico")
            outputs = self.model(preprocess_frames['pixel_values'].cuda())
            #print("predetto")
            '''
                Partiamo da immagini 128x128x3. Lui fa il resize a 224x224x3, quindi usando un tubelet 2x16x16 prende due frame su un patch 16x16.
                Quindi alla fine avremo che da 32 frame si riducono a 16 frame dato che il tubelet temporalmente crea patch ogni 2 frame, e in totale
                avremo (224/16)**2 patch ogni 2 frame, quindi (224/16)**2*(32/2) patch totali.
            '''
            #[1, 3137, 768]
            last_hidden_states = outputs.last_hidden_state
            '''
                Riduco 768 a 729 cosi che ogni patch (patch di due frame) posso trasformarlo in un cubo 9x9x9.
                Per ogni coppia di immagini sono stati presi 14x14=196 patch (dove 14=(224/16)). Quindi per ogni coppia di
                immagini reshapo le patch in (14*9,14*9, 9) = (9, 126, 126). Avendo 32/2 = 16 coppie di frame analizzate, 
                il tensore finale sarà (144, 126, 126) (dove 144=16*9).

                Devo ridurre il tensore in (64, 64, 64). Quindi faccio un downsampling.
                
            '''
            #tolgo il cls
            frames_hidden_states = last_hidden_states[0,1:,:]
            resized_embedding = nn.functional.interpolate(frames_hidden_states.unsqueeze(0), (729)).squeeze(0)
            #(144, 126, 126)
            reverse = self.reverse_patching(resized_embedding)
            #downsampling
            #(1,64,64,64)
            output = torch.nn.functional.interpolate(reverse.unsqueeze(0).unsqueeze(0), (64,64,64)).squeeze(0) if output is None else torch.cat([output, torch.nn.functional.interpolate(reverse.unsqueeze(0).unsqueeze(0), (64,64,64)).squeeze(0)], dim=0) 

    
        #print(output.shape)
        return output.unsqueeze(1)


if __name__ == '__main__':

    image_size = 64
    number_of_couples = 32
    
    #cond1 = torch.randn((2,1,32,32,32), dtype=torch.float32)
    sdf = torch.randn((2,1,image_size,image_size,image_size), dtype=torch.float32)
    frames = torch.randint(0, 255, (2,1,number_of_couples*4,image_size,image_size), dtype=torch.float32)

    model = FRAMES_VisionTransformer()

    print(model(frames).shape)