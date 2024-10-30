
import sys
import os

# Ottieni il percorso della directory del pacchetto principale
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Aggiungi il percorso alla lista dei percorsi di ricerca di Python
sys.path.append(package_path)

from pathlib import Path
import pandas as pd
import torch as th
import torch.nn.functional as F
import numpy as np
import yaml
from easydict import EasyDict
from PIL import Image
from src.utils import instantiate_from_config
from src.utils.vis import save_sdf_as_mesh

#::::::::::::::::::::::::::::::::::: GPU :::::::::::::::::::::::::::::::::::::::::::::::::::::

#scelgo la gpu
device = 0
th.set_grad_enabled(False)
th.cuda.set_device(device)

#::::::::::::::::::::::::::::::::::: CARICA FILE DI CONFIGURAZIONE E CHECKPOINT :::::::::::::::::::::::::::::::::::::::::::::::::::::


gen32_args_path = "/home/giuliofederico/raid/results2/new_dataset/240322_182442_towns/args.yaml"
gen32_ckpt_path = "/home/giuliofederico/raid/results2/new_dataset/240322_182442_towns/best_ep1248.pth"
sr64_args_path = "/home/giuliofederico/raid/results2/new_dataset_superresolution/240326_004845_towns/args.yaml"
sr64_ckpt_path = "/home/giuliofederico/raid/results2/new_dataset_superresolution/240326_004845_towns/best_ep1193.pth"

with open(gen32_args_path) as f:
    args1 = EasyDict(yaml.safe_load(f))
with open(sr64_args_path) as f:
    args2 = EasyDict(yaml.safe_load(f))


#:::::::::::::::::::::::::::::::::: CARICO MODELLO CHE DAI FRAME E DALLA SDF PRODUCE SDF :::::::::::::::::::::::::::::::::::::::::::::

#carico modello
model1 = instantiate_from_config(args1.model).cuda() #carica unet_sr3
ckpt = th.load(gen32_ckpt_path, map_location="cpu") #carica checkpoint
model1.load_state_dict(ckpt["model"])

#carico sampler
ddpm_sampler1 = instantiate_from_config(args1.ddpm.valid).cuda()

#carico preprocessor
preprocessor1 = instantiate_from_config(args1.preprocessor, "cuda")



#:::::::::::::::::::::::::::::::::: CARICO MODELLO CHE DALLA SDF PRODUCE SDF :::::::::::::::::::::::::::::::::::::::::::::

#carico modello
model2 = instantiate_from_config(args2.model).cuda() #crea la Unet Resize
ckpt = th.load(sr64_ckpt_path, map_location="cpu") #carica checkpoint
model2.load_state_dict(ckpt["model"])

#carico sampler
ddpm_sampler2 = instantiate_from_config(args2.ddpm.valid).cuda()

#carico preprocessor
preprocessor2 = instantiate_from_config(args2.preprocessor, "cuda")



#::::::::::::::::::::::::::::::::::::::: Carico i database originali di training e validazione ::::::::::::::::::::::::::::::

number_of_couples = 8 #lasciare cosi in quanto è stato allenato con 16

#------------------- stage 1 ---------------------
#TRAIN
annotated_db_grt_train = pd.read_pickle("/home/giuliofederico/SDF-Diffusion_x32_conditioned/scripts/annotated_db_grt_train.pkl")
annotated_db_condition1_train = pd.read_pickle("/home/giuliofederico/SDF-Diffusion_x32_conditioned/scripts/annotated_db_condition1_train.pkl")
annotated_db_condition2_train = pd.read_pickle("/home/giuliofederico/SDF-Diffusion_x32_conditioned/scripts/annotated_db_condition2_train.pkl")

#VALIDAZIONE
annotated_db_grt_val = pd.read_pickle("/home/giuliofederico/SDF-Diffusion_x32_conditioned/scripts/annotated_db_grt_val.pkl")
annotated_db_condition1_val = pd.read_pickle("/home/giuliofederico/SDF-Diffusion_x32_conditioned/scripts/annotated_db_condition1_val.pkl")
annotated_db_condition2_val = pd.read_pickle("/home/giuliofederico/SDF-Diffusion_x32_conditioned/scripts/annotated_db_condition2_val.pkl")


#------------------ stage 2 -----------------------
#TRAIN
annotated_db_grt_train_stage2 = pd.read_pickle("/home/giuliofederico/SDF-Diffusion_x32_conditioned/scripts/stage_2_annotated_dataset_ground_truth_train.pkl")
annotated_db_condition1_train_stage2 = pd.read_pickle("/home/giuliofederico/SDF-Diffusion_x32_conditioned/scripts/stage_2_annotated_dataset_condition_1_train.pkl")
annotated_db_condition2_train_stage2 = pd.read_pickle("/home/giuliofederico/SDF-Diffusion_x32_conditioned/scripts/stage_2_annotated_dataset_condition_2_train.pkl")

#VALIDAZIONE
annotated_db_grt_val_stage2 = pd.read_pickle("/home/giuliofederico/SDF-Diffusion_x32_conditioned/scripts/stage_2_annotated_dataset_ground_truth_val.pkl")
annotated_db_condition1_val_stage2 = pd.read_pickle("/home/giuliofederico/SDF-Diffusion_x32_conditioned/scripts/stage_2_annotated_dataset_condition_1_val.pkl")
annotated_db_condition2_val_stage2 = pd.read_pickle("/home/giuliofederico/SDF-Diffusion_x32_conditioned/scripts/stage_2_annotated_dataset_condition_2_val.pkl")


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: Utils :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#ritorna il ground truth sdf, la condition sdf e i frames
def getitem(idx, annotated_db_grt, annotated_db_condition1, annotated_db_condition2):
        
        #(64,64,64)
        grt = np.load(annotated_db_grt.iloc[idx]["path"])['arr_0']
        # #(1, 64, 64, 64)
        grt = th.from_numpy(grt)[None]
        grt = th.nn.functional.interpolate(grt.unsqueeze(0), size=(32,32,32), mode='nearest').squeeze(0)

        #test resize
        #(64,64,64)
        #grt = np.load(self.annotated_db_grt.iloc[idx]["path"])['arr_0']
        #grt = self.crop_and_resize(sdf=grt)
        # grt = th.from_numpy(grt)[None]
        # grt = th.nn.functional.interpolate(grt.unsqueeze(0), size=(32,32,32), mode='nearest').squeeze(0)




        # #(64,64,64)
        condition1 = np.load(annotated_db_condition1.iloc[idx]["path"])['arr_0']
        # #(1, 64, 64, 64)
        condition1 = th.from_numpy(condition1)[None]
        condition1 = th.nn.functional.interpolate(condition1.unsqueeze(0), size=(32,32,32), mode='nearest').squeeze(0)

        #test resize
        #(64,64,64)
        # condition1 = np.load(self.annotated_db_condition1.iloc[idx]["path"])['arr_0']
        # condition1 = self.crop_and_resize(sdf=condition1)


        condition2 = th.zeros((1,4*number_of_couples,64,64),dtype=th.float32)

        i = 0
        for path in annotated_db_condition2.iloc[idx]["paths"]:

            #carico immagine
            img = Image.open(path)
            img = img.convert('RGB')
            img = np.array(img)
            img = th.tensor(np.transpose(img, (2, 0, 1)), dtype=th.float32)

            name = path.split("/")[-1]

            #se si tratta di color
            if "color" in name:
                #simple wise standardization
                #normalizzo canale rosso
                img[0,:,:] = img[0,:,:]/255
                #normalizzo canale verde
                img[1,:,:] = img[1,:,:]/255
                #normalizzo canale blu
                img[2,:,:] = img[2,:,:]/255

                condition2[:,i:i+3,:,:] = img
            else:
                #prendo e normalizzo solo canale rosso che contiene le classi
                condition2[:,i+3,:,:] = img[0,:,:]/255
                i = i + 4
        
        condition2 = th.nn.functional.interpolate(condition2.unsqueeze(0), size=(32,32,32), mode='nearest').squeeze(0)
        splitted = annotated_db_grt.iloc[idx]['path'].split("/")

        model_id = splitted[-4]+"_"+splitted[-3]

        return "town", model_id, grt, condition1, condition2, 0



#::::::::::::::::::::::::::::::::::::::::::::: STAGE 1: DATI I FRAME E L'SDF PRODUCI LA SDF ::::::::::::::::::::::::::::::::::::::::

# def get_noisy_grt(grt):
#     # Definisci le dimensioni del sottocubo
#     dim_x = 16
#     dim_y = 16
#     dim_z = 16

#     # Genera rumore gaussiano per il sottocubo
#     noise = np.random.normal(loc=0.0, scale=1.0, size=(1,dim_x, dim_y, dim_z))
#     noise = th.from_numpy(noise).unsqueeze(0)

#     start_x, start_y, start_z = np.random.randint(0,15), np.random.randint(0,15), np.random.randint(0,15)
#     end_x, end_y, end_z = start_x+dim_x-1, start_y+dim_y-1, start_z+dim_z-1

#     # Aggiungi il rumore al volume principale nel sottocubo specificato
#     grt[:,:,start_x:end_x+1, start_y:end_y+1, start_z:end_z+1] += noise

#     return grt


# #copia i db con cui vuoi lavorare, se con i training o con le validation
# annotated_db_grt = annotated_db_grt_val
# annotated_db_condition1 = annotated_db_condition1_val
# annotated_db_condition2 = annotated_db_condition2_val

# data_type = "shape_completion_stage_1_val"

# results = pd.DataFrame(columns=['name', 'loss'])

# #loss function
# loss_func = th.nn.L1Loss(reduction="mean").to(device)

# for i in np.arange(len(annotated_db_grt)):

#     print("Elaboro dato ", i+1)
#     #indice
#     index = i

#     #carico grt, condition sdf e frames
#     class_name, model_id, grt, condition_sdf, condition_frames, cls = getitem(index, annotated_db_grt, annotated_db_condition1, annotated_db_condition2)

#     #li porto nel formato (batch_size, ....)
#     grt = grt.unsqueeze(0)
#     condition_sdf = condition_sdf.unsqueeze(0).cuda()
#     condition_frames = condition_frames.unsqueeze(0).cuda()

#     #preprocesso condition sdf (i frame sono già stati normalizzati)
#     condition_sdf = preprocessor1.standardize(condition_sdf, 0)

#     if "shape_completion" in data_type:
#         grt_for_completion = get_noisy_grt(preprocessor1.standardize(grt, 1))
#         grt_for_completion = grt_for_completion.cuda()
#         #predico la grt shape completed
#         predicted_grt = ddpm_sampler1.sample_ddim(lambda x, t: model1(x, grt_for_completion, condition_frames, t), condition_sdf.shape, show_pbar=True)
#         grt_for_completion = preprocessor1.destandardize(grt_for_completion, 1)
#     else:
#         #predico la grt
#         predicted_grt = ddpm_sampler1.sample_ddim(lambda x, t: model1(x, condition_sdf, condition_frames, t), condition_sdf.shape, show_pbar=True)

#     #calcolo la loss
#     loss = loss_func(grt.cuda(), predicted_grt)

#     #la destandardizzo
#     predicted_grt = preprocessor1.destandardize(predicted_grt, 1)
#     #la destandardizzo
#     condition_sdf = preprocessor1.destandardize(condition_sdf, 0)

#     # #salvo la condition 
#     # for i, out in enumerate(condition_sdf):
#     #     save_sdf_as_mesh(f"/home/giuliofederico/SDF-Diffusion_x32_conditioned/scripts/results/{model_id}_{data_type}_condition.obj", out, safe=True)

#     if "shape_completion" in data_type:
#         #salvo la sdf reale
#         for i, out in enumerate(grt_for_completion):
#             save_sdf_as_mesh(f"/home/giuliofederico/SDF-Diffusion_x32_conditioned/scripts/results/{data_type}/{model_id}_{data_type}_noisy_grt.obj", out, safe=True)
#         for i, out in enumerate(grt):
#             save_sdf_as_mesh(f"/home/giuliofederico/SDF-Diffusion_x32_conditioned/scripts/results/{data_type}/{model_id}_{data_type}_grt.obj", out, safe=True)

#     #salvo la sdf predetta
#     for i, out in enumerate(predicted_grt):
#         save_sdf_as_mesh(f"/home/giuliofederico/SDF-Diffusion_x32_conditioned/scripts/results/{data_type}/{model_id}_{data_type}_predicted_grt.obj", out, safe=True)
    

#     results = results._append({'name': model_id, "loss":loss.cpu().numpy()}, ignore_index=True)
#     print(loss)

# results.to_pickle("/home/giuliofederico/SDF-Diffusion_x32_conditioned/scripts/results_"+data_type+".pkl")





#::::::::::::::::::::::::::::::::::::::::::::: STAGE 2: DATI LA SDF MIP PRODUCI LA SDF ::::::::::::::::::::::::::::::::::::::::

def get_noisy_grt(grt):
    # Definisci le dimensioni del sottocubo
    dim_x = 16
    dim_y = 16
    dim_z = 16

    # Genera rumore gaussiano per il sottocubo
    noise = np.random.normal(loc=0.0, scale=1.0, size=(1,dim_x, dim_y, dim_z))
    noise = th.from_numpy(noise).unsqueeze(0)

    start_x, start_y, start_z = np.random.randint(0,15), np.random.randint(0,15), np.random.randint(0,15)
    end_x, end_y, end_z = start_x+dim_x-1, start_y+dim_y-1, start_z+dim_z-1

    # Aggiungi il rumore al volume principale nel sottocubo specificato
    grt[:,:,start_x:end_x+1, start_y:end_y+1, start_z:end_z+1] += noise

    return grt

#copia i db con cui vuoi lavorare, se con i training o con le validation
annotated_db_grt = annotated_db_grt_val_stage2
annotated_db_condition1 = annotated_db_condition1_val_stage2
annotated_db_condition2 = annotated_db_condition2_val_stage2

data_type = "shape_completion_stage_2_val"

results = pd.DataFrame(columns=['name', 'loss'])

#loss function
loss_func = th.nn.L1Loss(reduction="mean").to(device)

for i in np.arange(len(annotated_db_grt)):

    print("Elaboro dato ", i+1)
    #indice
    index = i

    #carico grt, condition sdf e frames
    class_name, model_id, grt, condition_grt, condition_frames, cls = getitem(index, annotated_db_grt, annotated_db_condition1, annotated_db_condition2)

    #li porto nel formato (batch_size, ....)
    grt = grt.unsqueeze(0)
    condition_grt = condition_grt.unsqueeze(0).cuda()

    #preprocesso grt
    grt =  preprocessor2.standardize(grt, 1)
    #preprocesso condition grt
    condition_grt = preprocessor2.standardize(condition_grt, 0)

    if "shape_completion" in data_type:
        grt_for_completion = get_noisy_grt(condition_grt.cpu())
        grt_for_completion = grt_for_completion.cuda()
        #predico la grt shape completed
        predicted_grt = ddpm_sampler2.sample_ddim(lambda x, t: model2(x, grt_for_completion, t), condition_grt.shape, show_pbar=True)
        grt_for_completion = preprocessor2.destandardize(grt_for_completion, 0)
    else:
        #predico la grt
        predicted_grt = ddpm_sampler2.sample_ddim(lambda x, t: model2(x, condition_grt, t), condition_grt.shape, show_pbar=True)

    #calcolo la loss
    loss = loss_func(grt.cuda(), predicted_grt)

    #la destandardizzo
    predicted_grt = preprocessor2.destandardize(predicted_grt, 1)
    #la destandardizzo
    condition_grt = preprocessor2.destandardize(condition_grt, 0)
    grt = preprocessor2.destandardize(grt, 1)

    if "shape_completion" in data_type:
        #salvo la sdf reale
        for i, out in enumerate(grt_for_completion):
            save_sdf_as_mesh(f"/home/giuliofederico/SDF-Diffusion_x32_conditioned/scripts/results_stage2/{data_type}/{model_id}_{data_type}_noisy_grt.obj", out, safe=True)
        for i, out in enumerate(grt):
            save_sdf_as_mesh(f"/home/giuliofederico/SDF-Diffusion_x32_conditioned/scripts/results_stage2/{data_type}/{model_id}_{data_type}_grt.obj", out, safe=True)

    
    #salvo la sdf predetta
    for i, out in enumerate(predicted_grt):
        save_sdf_as_mesh(f"/home/giuliofederico/SDF-Diffusion_x32_conditioned/scripts/results_stage2/{data_type}/{model_id}_{data_type}_predicted_grt.obj", out, safe=True)
    

    results = results._append({'name': model_id, "loss":loss.cpu().numpy()}, ignore_index=True)
    print(loss)

results.to_pickle("/home/giuliofederico/SDF-Diffusion_x32_conditioned/scripts/results_"+data_type+".pkl")


















