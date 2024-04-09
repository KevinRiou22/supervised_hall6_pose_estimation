# Hall6 data
## downlod the data
create a folder named "data" in the root directory of the project, and download the data from the following link (dwd everything in the "data" folder):
https://uncloud.univ-nantes.fr/index.php/s/XP2cqPccBEdS3qC
## triangulate 3D poses from 2D poses
```bash
python triangulate_hall6.py --cfg ./cfg/submit/cfg_triangulate_16_cams.yaml  --triang_out_name triangulated_3D_16_cams
```
cfg : config file for triangulation, several triangulation config files are provided (cfg_triangulate_*.yaml)
triang_out_name: containing the triangulated 3D poses, as well as the input 2D poses used for triangulation, and "ground-truth" 2D poses obtained by reprojecting  the triangulated 3D poses to 2D.

## evaluate the triangulated 3D poses
```bash 
python evaluate_triangulation.py --cfg ./cfg/submit/cfg_triangulate_16_cams.yaml --data_name triangulated_3D_16_cams
``` 

## train 3D pose estimators
Baseline 1
```bash

```

Visualize the poses for the first example in the validation set, for epoch 0 :
```bash
python visualize_poses.py --visu_path ./visu/submit/baseline1/absolute/main/ --visu_epoch 0
```

Visualize the trajectory of the root, for the first example in the validation set, for epoch 0 (only for absolute pose estimators)) :
```bash 
python visualize_root_path.py --path ./visu/submit/baseline1/absolute/main/ --epoch 0
```