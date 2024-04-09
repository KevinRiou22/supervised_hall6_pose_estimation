# Hall6 data
## downlod the data
create a folder named "data" in the root directory of the project, and download the data from the following link (dwd everything in the "data" folder):
https://uncloud.univ-nantes.fr/index.php/s/XP2cqPccBEdS3qC
## install requirements
```bash
pip install -r requirements.txt
```
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

## Train the confidence prediction models
### create experiment dirs
```bash
sh create_exp_dirs.sh
```

### train the confidence prediction model
you can find the commands to launch the various trainings of the confidence prediction models in the folders
cfg/submit/conf_pred_*/

the respective commands are located at the end of the slurm_cmd files located in the above mentioned folders.


## train 3D pose estimators on the data (ongoing work)


### train the baseline 1
```bash
python run_hall6.py --cfg ./cfg/submit/baseline1/absolute/cfg.yaml  --ckpt_path ./checkpoint/submit/baseline1/absolute/main/ --log_path ./log/submit/baseline1/absolute/main/  --visu_path ./visu/submit/baseline1/absolute/main/
```

Visualize the poses for the first example in the validation set, for epoch 0 :
```bash
python visualize_poses.py --visu_path ./visu/submit/baseline1/absolute/main/ --visu_epoch 0
```

Visualize the trajectory of the root, for the first example in the validation set, for epoch 0 (only for absolute pose estimators)) :
```bash 
python visualize_root_path.py --path ./visu/submit/baseline1/absolute/main/ --epoch 0
```