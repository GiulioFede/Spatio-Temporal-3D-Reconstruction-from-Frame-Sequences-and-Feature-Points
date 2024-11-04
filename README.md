# Spatio-Temporal-3D-Reconstruction-from-Frame-Sequences-and-Feature-Points

Reconstructing a large real environment is a fundamental task to promote eXtended Reality adoption in industrial and entertainment fields. However, the short range of depth cameras, the sparsity of LiDAR sensors, and the huge computational cost of Structure-from-Motion pipelines prevent scene replication in near real time. To overcome these limitations, we introduce a spatio-temporal diffusion neural architecture, a generative AI technique that fuses temporal information (i.e., a short temporally-ordered list of color photographs, like sparse frames of a video stream) with an approximate spatial resemblance of the explored environment. Our aim is to modify an existing 3D diffusion neural model to produce a Signed Distance Field volume from which a 3D mesh representation can be extracted. 

## Dataset and config file

Each experiment will be saved in its own folder whose destination will be indicated in the key `exp_dir`

The config file is located in `src/towns`. In the key `dataset->params->ds_kwargs->path_to_db`, the path to the main folder containing the dataset must be included. The dataset itself must be organized as follows:
```
.
├── Town01
│   ├── 0
│   │   ├── capture
│   │   │   ├── 0_color.png
│   │   │   ├── 0_sseg.png
│   │   │   ├── 1_color.png
│   │   │   ├── 1_sseg.png
.   .   .   .
.   .   .   .
│   │   │   ├── 59_color.png
│   │   │   ├── 59_sseg.png
│   │   └── output
│   │       ├── 04_scene_sdf.npz     <---- ground truth SDF
│   │       ├── 10_slam_sdf.npz      <---- coarse SDF
│   ├── 1
│   │   ├── capture
│   │   │   ├── 0_color.png
│   │   │   ├── 0_sseg.png
│   │   │   ├── 1_color.png
│   │   │   ├── 1_sseg.png
.   .   .   .
.   .   .   .
│   │   │   ├── 59_color.png
│   │   │   ├── 59_sseg.png
│   │   └── output
│   │       ├── 04_scene_sdf.npz    
│   │       ├── 10_slam_sdf.npz 
.   .   .
.   .   .        
├── Town02
│   ├── 0
│   │   ├── capture
│   │   │   ├── 0_color.png
│   │   │   ├── 0_sseg.png
│   │   │   ├── 1_color.png
│   │   │   ├── 1_sseg.png
.   .   .   .
.   .   .   .
│   │   │   ├── 59_color.png
│   │   │   ├── 59_sseg.png
│   │   └── output
│   │       ├── 04_scene_sdf.npz     
│   │       ├── 10_slam_sdf.npz      
```

The dataset must be structured in such a way that at the first level there are folders with the names of the Towns to be modeled. Inside each there must be N folders numbered with integers, each indicating for example a "district" of the city to be modeled. Finally each of these folders has two more: *capture* containing 60 temporal pairs of RGB frames and semantic map, *output* containing the SDF (DxDxD) coarse (10_slam_sdf.npz) and the ground truth (04_scene_sdf.npz).

> [!NOTE]
> Currently the file configurations are designed to handle 64x64x64 volumes.

The key `dataset->params->ds_kwargs->number_of_couples` specifies the number of pairs (rgb and semantic map) to consider. If there are N pairs in total in the dataset, then the _number_of_couples_ will be taken uniformly along the entire path.

The keys `preprocessor->params->maxs`, `preprocessor->params->mins`, `preprocessor->params->sdf_clip`, `preprocessor->params->mean` and `preprocessor->params->std` must contain the statistics of the dataset. In particular, in each array the first element refers to the coarse statistics, the second to the ground truth. Currently in the file `models->trainers->sr3d.py` only the standardization with previous truncation is implemented.


## Training
Our architecture is two-stage. In the first stage, coarse volumes, in the form of MIP_SDF, are transformed into detailed MI_SDF volumes. In the second, MIP_SDF volumes are transformed back into SDF. Currently, there is a single command to start training the first stage.

```
python main.py src/config/tows.py
```
If you need to use MIP_SDF (vital when the data has a very unbalanced distribution of negative and positive values) consider creating a dataset where each volume is modified by taking its bounding box and enlarging the object until it adheres to the 64x64x64 volume. Train the first stage with this new dataset and the second using the volumes of the new dataset as input and the old dataset as ground truth. Consider modifying the `dataset.py` file so that in the get_item only the volume is returned instead of the volume and the frames.
An example of the overall pipeline (for inference ) is in the file scripts/test.py.

## Citation

```
@inproceedings{10.1145/3672406.3672415,
author = {Federico Giulio, Carrara Fabio, Amato Giuseppe and Di Benedetto Marco},
title = {Spatio-Temporal 3D Reconstruction from Frame Sequences and Feature Points},
year = {2024},
isbn = {9798400717949},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3672406.3672415},
doi = {10.1145/3672406.3672415},
abstract = {Reconstructing a large real environment is a fundamental task to promote eXtended Reality adoption in industrial and entertainment fields. However, the short range of depth cameras, the sparsity of LiDAR sensors, and the huge computational cost of Structure-from-Motion pipelines prevent scene replication in near real time. To overcome these limitations, we introduce a spatio-temporal diffusion neural architecture, a generative AI technique that fuses temporal information (i.e., a short temporally-ordered list of color photographs, like sparse frames of a video stream) with an approximate spatial resemblance of the explored environment. Our aim is to modify an existing 3D diffusion neural model to produce a Signed Distance Field volume from which a 3D mesh representation can be extracted. Our results show that the hallucination approach of diffusion models is an effective methodology where a fast reconstruction is a crucial target.},
booktitle = {Proceedings of the 2024 ACM International Conference on Interactive Media Experiences Workshops},
pages = {52–64},
numpages = {13},
keywords = {3D Reconstruction, Artificial Intelligence, Deep Learning, Denoising Diffusion Probabilistic Model, Machine Learning, Signed Distance Field, Video Reconstruction},
location = {Stockholm, Sweden},
series = {IMXw '24}
}
