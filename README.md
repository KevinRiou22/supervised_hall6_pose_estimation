# Hall6 data
## downlod the data
create a folder named "data" in the root directory of the project, and download the data from the following link (dwd everything in the "data" folder):
https://uncloud.univ-nantes.fr/index.php/s/XP2cqPccBEdS3qC
## install requirements
```bash
pip install -r requirements.txt
```
## triangulate 3D poses from 2D poses
Here is how to use the code to triangulate the 2D poses to 3D (weighted using confidences of 2D pose estimator as weight, the next section will detail how to get our dynamic weights)
If you want to access the already triangulated poses, download the triangulate_16_cams.npz on our data link.
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
Here is how to use the code to train the dynamic weights prediction models.
If you want to use the the already triangulated poses (using dyn weights), download the data_epoch_59_16_cams.npz on our data link.

you can find the commands to launch the various trainings of the confidence prediction models in the folders
cfg/submit/conf_pred_*/

the respective commands are located at the end of the slurm_cmd files located in the above mentioned folders.


## train 3D pose estimators on the data (ongoing work)


### train the baseline 1
```bash
python run_hall6.py --cfg ./cfg/submit/baseline1/absolute/cfg.yaml  --ckpt_path ./checkpoint/submit/baseline1/absolute/main/ --log_path ./log/submit/baseline1/absolute/main/  --visu_path ./visu/submit/baseline1/absolute/main/ --data_name data_epoch_59_16_cams

```
the --data_name option allows to choose the data you want to use to train your models. For e.g., it can be triangulate_16_cams if you want to use the 3D poses obtained after weighted triangulation, using the confidences of the 2D pose estimator as weights. 
We recommand using the data_epoch_59_16_cams that provides 3D poses obtained using the dynamic weights obtained using our confidence prediction models.


Visualize the poses for the first example in the validation set, for epoch 0 :
```bash
python visualize_poses.py --visu_path ./visu/submit/baseline1/absolute/main/ --visu_epoch 0
```

Visualize the trajectory of the root, for the first example in the validation set, for epoch 0 (only for absolute pose estimators)) :
```bash 
python visualize_root_path.py --path ./visu/submit/baseline1/absolute/main/ --epoch 0
```

## Annotate the trajectories 
We also provide the annotation tool that allowed to annotate sub-trajectories with ergonomic labels.

To use this tool, launch the following command:
```bash 
    python annotate_trajectories.py --path_poses ./data/ --poses_filename  triangulate_16_cams.npz --subject 1 --task 1 --example 1 --path_labels ./data/
```
This tool is used to annotate the actions in the videos.
Since the ergonomic variations where all planned in advance for each participants, the matching of the actions with their ergonomic variations is automatic once the actions are annotated.

subject can have 1-5 values
tasks can have 1-3
for task 1, examples can be 1-15
for tasks 2 and 3, examples can be 1-6

Use the right arrow key to move forward in the trajectory, and left to move backwards.
The terminal displays the primitives that you need to annotate, ordered by timestamp.
So copy the first primitive name of from the list displayed in the terminal, search for it in the trajectory using arrows, and once you found it, past the primitive name in the annotation field and click on "save the label">
Note that the list of primitive contains "start_*" and "end_*" primitives for each action. 
Each time you annotated both of them, the frames in between are automatically annotated. You can verify it by playing with the arrows.
You can press esc whenever you want to quit, the labels are auto-saved continuously.
If you want to remove an annotation, go to the start of end frame (depending on which one you want to modify), empty the annotation text field, and click on "save label">


Once you annotated a video, you can check for the correct matching of the actions with the ergonomic variations with the following command.

```bash 
python visualize_ergo_labels.py --path_poses ./data/ --poses_filename  triangulate_16_cams.npz --subject 1 --task 1 --example 1 --path_labels ./data/
```

Note that their might be no labels for the first frames if the start of the first action was not defined in the first frame of the video.
