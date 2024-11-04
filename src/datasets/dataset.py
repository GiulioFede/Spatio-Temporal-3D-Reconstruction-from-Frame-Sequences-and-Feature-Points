from collections import defaultdict
import numpy as np
import pandas as pd
import os
import h5py
import torch as th
from src import logging
from PIL import Image
from src.utils import instantiate_from_config
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
import open3d as o3d


synset_to_cls = {
    "town": 0
}

cls_to_synset = {
    0: "town"
}

taxonomy_to_synset = {
    "town": "town"
}

synset_to_taxonomy = {
    "town": "town",
}



def create_annotated_dataset(data_root, k):

    spwan_points_to_remove = [  "Town01__44",     #(punti sparsi)
                                "Town01_Opt__54", #(punti sparsi)
                                "Town10HD__28",  #(gigante)
                                "Town10HD__29",  #(gigante)
                                "Town10HD_Opt__30",  #(gigante)
                                "Town10HD_Opt__31",  #(gigante)
                             ]


    #creo dataframe contenente per ogni riga i path verso le k coppie + volume input + volume ground truth
    annotated_dataset_1 = pd.DataFrame(columns=['path'])
    annotated_dataset_2 = pd.DataFrame(columns=['path'])
    annotated_dataset_3 = pd.DataFrame(columns=['paths'])

    choosen_couples_indexes = np.linspace(0, 59, k, dtype=int)

    
    #Mi posiziono nella cartella delle Town. Per ogni Town...
    for town_folder in os.listdir(data_root):
        
        #prendo il path verso la Town i-esima
        town_path = os.path.join(data_root, town_folder)

        # if "Opt" in town_path:
        #     continue

        #per ogni sua sottocartella (spawn points 0,1,2...)
        for spawn_point_folder in os.listdir(town_path):

            spawn_point_path = os.path.join(town_path, spawn_point_folder)

            '''
                Controllo se devo prendere tale spawn point secondo la seguente regola: controllo se esiste un altra condition
                che ha una Chamfer distance <= t1 (quindi molto simili) e una Chamfer distance tra i rispettivi ground truth >=t2 (quindi molto diversi).
                Se è cosi allora non dovrò tenere in considerazione tale mesh in quanto destabilizza il training, ossia quando alla Unet gli do A e B, con 
                A e B molto simili, le dico di predirre C e D con C e D molto diversi. 
                Ovviamente quando non considero lo spawn point andrò a eliminare ogni suo riferimento dal dataframe:
                                mesh1: spawn_point_corrente    mesh2: xxx     <----- rimuovo
                Però dovrò anche rimuovere tutte le coppie:
                                mesh1: xxx    mesh2: spawn_point_corrente     <----- rimuovo
                In quanto quando elaborerò nel ciclo di questo codice lo spawn_point xxx dovrò stavolta prenderlo in quanto il suo "simile" non è stato considerato.
            '''
            src_name = town_folder+"__"+spawn_point_folder
            # if src_name in spwan_points_to_remove:
            #     continue

            # check, df = check_spawn_point(df, src_name)

            # if check==True:
            #     continue
            
            #array contenente i path delle coppie
            couples_array = np.array([])
            #per ogni indice, carico rgb (i_color.png), semantic segmentation (i_sseg.png)
            for couple_index in choosen_couples_indexes:
                #carico rgb
                couples_array = np.append(couples_array,os.path.join(spawn_point_path,"capture",str(couple_index)+"_color.png"))
                couples_array = np.append(couples_array,os.path.join(spawn_point_path,"capture",str(couple_index)+"_sseg.png"))

                #test
                # couples_array = np.append(couples_array,os.path.join(spawn_point_path,"capture",str(couple_index)+"_color.png"))
                # couples_array = np.append(couples_array,os.path.join(spawn_point_path,"capture",str(couple_index)+"_depth.png"))
                # couples_array = np.append(couples_array,os.path.join(spawn_point_path,"capture",str(couple_index)+"_iseg.png"))
                # couples_array = np.append(couples_array,os.path.join(spawn_point_path,"capture",str(couple_index)+"_normal.png"))
                # couples_array = np.append(couples_array,os.path.join(spawn_point_path,"capture",str(couple_index)+"_sseg_colored.png"))
                # couples_array = np.append(couples_array,os.path.join(spawn_point_path,"capture",str(couple_index)+"_sseg.png"))
            
            #aggiungo riga al dataframe
            
            annotated_dataset_1 = annotated_dataset_1._append({'path': os.path.join(spawn_point_path,"output","04_scene_sdf_scaled.npz")}, ignore_index=True)
            annotated_dataset_2 = annotated_dataset_2._append({'path': os.path.join(spawn_point_path,"output","10_slam_sdf_scaled.npz")}, ignore_index=True)
            
            # annotated_dataset_1 = annotated_dataset_1._append({'path': os.path.join(spawn_point_path,"output","04_scene_sdf.npz")}, ignore_index=True)
            # annotated_dataset_2 = annotated_dataset_2._append({'path': os.path.join(spawn_point_path,"output","04_scene_sdf_scaled.npz")}, ignore_index=True)

            annotated_dataset_3 = annotated_dataset_3._append({'paths': couples_array}, ignore_index=True)
            
            
    return [annotated_dataset_1, annotated_dataset_2, annotated_dataset_3]


def crop_and_resize(sdf):

        indices = np.where(sdf <= 2)
        # Determina il minimo e il massimo delle coordinate lungo ciascuna dimensione
        min_x, max_x = min(indices[0]), max(indices[0])
        min_y, max_y = min(indices[1]), max(indices[1])
        min_z, max_z = min(indices[2]), max(indices[2])
            
        # Costruisci un nuovo cubo utilizzando le coordinate minime e massime
        sub_v = sdf[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]

        f = (sub_v[sub_v<=0].flatten().std())/(sub_v.flatten().std())
        sub_v[sub_v>=0] = sub_v[sub_v>=0]*f
        

        #resize
        sub_v = th.from_numpy(sub_v)

        #mip-mapping
        #metti linear

        sub_v = th.nn.functional.interpolate(sub_v.unsqueeze(0).unsqueeze(0), size=(32,32,32), mode='linear').squeeze(0)

        return sub_v




def compute_mean_std(db, what, mean):

    x = 0
    N = 0

    num_data = len(db)

    for idx in np.arange(num_data):

        print("elaboro idx:", idx)

        path_sdf = db.iloc[idx]["path"]
        sdf = np.load(path_sdf)['arr_0']

        #sdf = crop_and_resize(sdf)
        sdf = th.nn.functional.interpolate(th.from_numpy(sdf).unsqueeze(0).unsqueeze(0), size=(32,32,32), mode='nearest').squeeze(0)

        if what=="mean":
            x += sdf.numpy().flatten().sum()
        else:
            x += ((sdf.numpy().flatten()-mean)**2).sum()

        N += len(sdf.numpy().flatten())
    
    if what == "mean":
        return x/N
    else:
        return np.sqrt(x/N)





class DatasetSR(Dataset):
    def __init__(self, path_to_db, number_of_couples, cates, image_size, split) -> None:
        super().__init__()

        self.path_to_db = path_to_db
        self.number_of_couples = number_of_couples
        self.seed = 4
        self.train_val_split = 0.15
        self.split = split
        self.image_size = image_size

        self.annotated_db_grt, self.annotated_db_condition1, self.annotated_db_condition2 = self.get_db()

        self.counter = defaultdict(int)
        self.cate_indices = defaultdict(list)

        self.counter['town'] = len(self.annotated_db_grt)
        self.cate_indices['town'] = np.arange(self.counter['town'])
        self.synset_to_cls = synset_to_cls
        self.cls_to_synset = cls_to_synset
        self.cates = taxonomy_to_synset[cates]
        self.n_classes = len(self.synset_to_cls)


   


    def get_db(self):

        np.random.seed(self.seed)
        annotated_dataset_ground_truth, annotated_dataset_condition_1, annotated_dataset_condition_2 = create_annotated_dataset(data_root=self.path_to_db,
        
                                                                                                                           k=self.number_of_couples)
        
        # mean_grt = compute_mean_std(annotated_dataset_ground_truth, "mean", None)
        # std_grt = compute_mean_std(annotated_dataset_ground_truth, "std", mean_grt)
        # mean_cond1 = compute_mean_std(annotated_dataset_condition_1, "mean", None)
        # std_cond1 = compute_mean_std(annotated_dataset_condition_1, "std", mean_cond1)



        # Crea una sequenza di indici per il shuffle
        shuffle_indexes = np.random.permutation(len(annotated_dataset_ground_truth))
        # Applico la sequenza di indici a entrambi i dataset

        annotated_dataset_ground_truth = annotated_dataset_ground_truth.iloc[shuffle_indexes].reset_index(drop=True)
        annotated_dataset_condition_1 = annotated_dataset_condition_1.iloc[shuffle_indexes].reset_index(drop=True)
        annotated_dataset_condition_2 = annotated_dataset_condition_2.iloc[shuffle_indexes].reset_index(drop=True)

        #divido i 3 dataset in training e validation
        # Calcolo l'indice per dividere il DataFrame
        indice_split = int((1-self.train_val_split) * len(annotated_dataset_ground_truth))

        # Suddividi il DataFrame in set di addestramento e validazione
        annotated_dataset_ground_truth_train = annotated_dataset_ground_truth.iloc[:indice_split]
        annotated_dataset_ground_truth_val = annotated_dataset_ground_truth.iloc[indice_split:]

        annotated_dataset_condition_1_train = annotated_dataset_condition_1.iloc[:indice_split]
        annotated_dataset_condition_1_val = annotated_dataset_condition_1.iloc[indice_split:]

        annotated_dataset_condition_2_train = annotated_dataset_condition_2.iloc[:indice_split]
        annotated_dataset_condition_2_val = annotated_dataset_condition_2.iloc[indice_split:]

        if self.split == "train":
            return [annotated_dataset_ground_truth_train, annotated_dataset_condition_1_train,annotated_dataset_condition_2_train]
        else:
            return [annotated_dataset_ground_truth_val, annotated_dataset_condition_1_val, annotated_dataset_condition_2_val]
        


    def __len__(self):
        return len(self.annotated_db_grt)
    

    def crop_and_resize(self,sdf):

        indices = np.where(sdf <= 1)
        # Determina il minimo e il massimo delle coordinate lungo ciascuna dimensione
        min_x, max_x = min(indices[0]), max(indices[0])
        min_y, max_y = min(indices[1]), max(indices[1])
        min_z, max_z = min(indices[2]), max(indices[2])
            
        # Costruisci un nuovo cubo utilizzando le coordinate minime e massime
        sub_v = sdf[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]

        f = (sub_v[sub_v<=0].flatten().std())/(sub_v.flatten().std())
        sub_v[sub_v>=0] = sub_v[sub_v>=0]*f
        

        #resize
        sub_v = th.from_numpy(sub_v)

        sub_v = th.nn.functional.interpolate(sub_v.unsqueeze(0).unsqueeze(0), size=(32,32,32), mode='nearest').squeeze(0)

        return sub_v

    
    # def crop_and_resize(self, sdf):

    #     indices = np.where(sdf <= 2)
    #     # Determina il minimo e il massimo delle coordinate lungo ciascuna dimensione
    #     min_x, max_x = min(indices[0]), max(indices[0])
    #     min_y, max_y = min(indices[1]), max(indices[1])
    #     min_z, max_z = min(indices[2]), max(indices[2])
            
    #     # Costruisci un nuovo cubo utilizzando le coordinate minime e massime
    #     sub_v = sdf[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]

    #     #resize
    #     sub_v = th.from_numpy(sub_v)
    #     #sub_v = th.nn.functional.interpolate(sub_v.unsqueeze(0).unsqueeze(0), size=(64,64,64), mode='nearest')

    #     sub_v = th.nn.functional.interpolate(sub_v.unsqueeze(0).unsqueeze(0), size=(32,32,32), mode='nearest').squeeze(0)

    #     return sub_v
    


    # def crop_and_resize(self, sdf):

    #     indices = np.where(sdf <= 2)
    #     # Determina il minimo e il massimo delle coordinate lungo ciascuna dimensione
    #     min_x, max_x = min(indices[0]), max(indices[0])
    #     min_y, max_y = min(indices[1]), max(indices[1])
    #     min_z, max_z = min(indices[2]), max(indices[2])
            
    #     # Costruisci un nuovo cubo utilizzando le coordinate minime e massime
    #     sub_v = sdf[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]

    #     f = (sub_v[sub_v<=0].flatten().std())/(sub_v.flatten().std())
    #     sub_v[sub_v>=0] = sub_v[sub_v>=0]*f
        

    #     #resize
    #     sub_v = th.from_numpy(sub_v)

    #     sub_v = th.nn.functional.interpolate(sub_v.unsqueeze(0).unsqueeze(0), size=(32,32,32), mode='nearest').squeeze(0)

    #     return sub_v





    '''
        Ritorna una sdf:
            1) sdf_y: la 64x64x64
          
        
            Prima di ritornala la trasforma in (1, x,x,x)
    '''
    def __getitem__(self, idx):
        
        print("")

        #(64,64,64)
        grt = np.load(self.annotated_db_grt.iloc[idx]["path"])['arr_0']
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
        condition1 = np.load(self.annotated_db_condition1.iloc[idx]["path"])['arr_0']
        # #(1, 64, 64, 64)
        condition1 = th.from_numpy(condition1)[None]
        condition1 = th.nn.functional.interpolate(condition1.unsqueeze(0), size=(32,32,32), mode='nearest').squeeze(0)

        #test resize
        #(64,64,64)
        # condition1 = np.load(self.annotated_db_condition1.iloc[idx]["path"])['arr_0']
        # condition1 = self.crop_and_resize(sdf=condition1)


        condition2 = th.zeros((1,4*self.number_of_couples,64,64),dtype=th.float32)

        i = 0
        for path in self.annotated_db_condition2.iloc[idx]["paths"]:

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
        splitted = self.annotated_db_grt.iloc[idx]['path'].split("/")

        model_id = splitted[-4]+"_"+splitted[-3]

        return "town", model_id, grt, condition1, condition2, 0
    

    def get_sample_idx(self, n_samples_per_cates):
        n = self.n_classes * n_samples_per_cates

        sample_idx = []
        for synset in self.synset_to_cls:
            sample_idx += self.cate_indices[synset][:n]

        return sample_idx


def build_dataloaders(ddp, ds_kwargs, dl_kwargs):
    splits = ("train", "val", "test")
    #per train, val e test costruisce DatasetSR
    dss = [DatasetSR(split=split, **ds_kwargs) for split in splits]

    log = logging.getLogger()
    log.info("Dataset Loaded:")
    for synset in dss[0].counter.keys():
        msg = f"    {synset} {synset_to_taxonomy[synset]:20}"
        msg += f" {dss[0].counter[synset]:5} {dss[1].counter[synset]:5} {dss[2].counter[synset]:5}"
        log.info(msg)

    tff = [True, False, False]
    if ddp:
        samplers = [DistributedSampler(ds, shuffle=t) for ds, t in zip(dss, tff)]
        dls = [DataLoader(ds, **dl_kwargs, sampler=sampler) for ds, sampler in zip(dss, samplers)]
    else:
        #ancora per train, val e test costruisce i rispettivi dataloader
        dls = [DataLoader(ds, **dl_kwargs, shuffle=t) for ds, t in zip(dss, tff)]

    return dls


def __test__():
    opt = """
target: src.datasets.dogn_sr.build_dataloaders
params:
    ds_kwargs:
        datafile_lr: /dev/shm/jh/data/sdf.res32.level0.0500.PC15000.pad0.20.hdf5
        datafile_hr: /dev/shm/jh/data/sdf.res64.level0.0313.PC15000.pad0.20.hdf5
        cates: all
    dl_kwargs:
        batch_size: 4
        num_workers: 0
        pin_memory: no
        persistent_workers: no
    """
    import yaml
    from src.utils import instantiate_from_config

    opt = yaml.safe_load(opt)
    dls = instantiate_from_config(opt, False)
    for synset, model_id, sdf_y, sdf_x, cls in dls[0]:
        break

    print(synset, model_id, sdf_y.shape, sdf_x.shape, cls)
    """
[22:09:13 14:03:49  INFO] Dataset Loaded:
[22:09:13 14:03:49  INFO]     02691156 airplane              2832   404   809
[22:09:13 14:03:49  INFO]     02828884 bench                 1272   181   363
[22:09:13 14:03:49  INFO]     02933112 cabinet               1101   157   281
[22:09:13 14:03:49  INFO]     02958343 car                   4911   749  1499
[22:09:13 14:03:49  INFO]     03001627 chair                 4746   677  1355
[22:09:13 14:03:49  INFO]     03211117 display                767   109   219
[22:09:13 14:03:49  INFO]     03636649 lamp                  1624   231   463
[22:09:13 14:03:49  INFO]     03691459 loudspeaker           1134   161   323
[22:09:13 14:03:49  INFO]     04090263 rifle                 1661   237   474
[22:09:13 14:03:49  INFO]     04256520 sofa                  2222   317   634
[22:09:13 14:03:49  INFO]     04379243 table                 5958   850  1701
[22:09:13 14:03:49  INFO]     04401088 telephone              737   105   210
[22:09:13 14:03:49  INFO]     04530566 vessel                1359   193   387
('03001627', '04379243', '03211117', '04379243')
('4a0b61d33846824ab1f04c301b6ccc90', '441e0682fa5eea135c49e0733c4459d0',
 '2c4bcdc965d6de30cfe893744630a6b9', '1ab95754a8af2257ad75d368738e0b47')
torch.Size([4, 1, 64, 64, 64]) torch.Size([4, 1, 32, 32, 32]) tensor([0, 0, 4, 1])
    """


if __name__ == "__main__":
    __test__()
