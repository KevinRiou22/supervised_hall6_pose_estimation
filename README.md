# Hall6 data
## downlod the data

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

Baseline 2
```bash

```

Baseline 3
```bash

```

Baseline 4
```bash

```

Baseline 5
```bash

```

Investiguation 1
```bash

```

Investiguation 2
```bash

```

Investiguation 3
```bash

```

Investiguation 4
```bash

```

Investiguation 5
```bash

```

Investiguation 6
```bash

``` 

Investiguation 7
```bash

```
