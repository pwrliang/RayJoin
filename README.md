# RayJoin: Real-Time Spatial Join Processing with Ray-Tracing

## 1. Build

### 1.1 Install Dependencies
(1) [gflags](https://github.com/gflags/gflags)

(2) [glog](https://github.com/google/glog)

(3) [NVIDIA Optix 7.7+](https://developer.nvidia.com/designworks/optix/download)

(4) [NVIDIA CUDA 11.6+](https://developer.nvidia.com/cuda-11-6-0-download-archive)

(5) [NVIDIA Driver 530+](https://www.nvidia.com/download/index.aspx)

Install gflags and glog:
`sudo apt install libgflags-dev libgoogle-glog-dev`

Install Optix: 
```shell
export OPTIX_HOME=~/optix

mkdir -p $OPTIX_HOME
./NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64.sh --prefix=$OPTIX_HOME --exclude-subdir --skip-license
```

### 1.2 Building Instructions

- Debug (Building the project under the debug mode enables some counter to profile RayJoin)
```shell
mkdir Debug
cd Debug
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=$OPTIX_HOME ..
```

- Release
```shell
mkdir Build
cd Build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$OPTIX_HOME ..
```

## 2. Dataset Preparation

We use a format called CDB, which is described in this [paper](https://dl.acm.org/doi/abs/10.1145/2835185.2835188).
This format allows polygons storing in chains to save space. The chain also carries neighboring information of polygons, 
which makes point-in-polygon (PIP) test easier.
```
<chain id> <number of points in the chain> <first point id> <last point id> <left face id> <right face id>
<point coordinates>
```
Example:
```text
1 2 0 1 1 8
-3.6580000000e+01 -4.6636000000e+00
-3.6094300000e+01 -1.3593000000e+00
2 2 0 16657 925 4
-3.6580000000e+01 -4.6636000000e+00
-3.6594300000e+01 -4.8691000000e+00
```

### 2.1 Download Datasets
We cannot find a place that allows us to upload large datasets and hide our identity as well. Currently, we only provide a minimum dataset that verify the correctness of the program, which is under `test/maps`.
You can also process your own datasets with the script we provided under `misc/shp2cdb.py`

### 2.2 Process dataset with ArcGIS
We basically need two 4 steps to process the datasets, which can be found in the toolbox in ArcGIS Pro. 
1. Polygon To Line. Make sure "Identify and store polygon neighboring information" is checked.
2. Feature Vertices To Points, which converts the output from step 1 to points.
3. Add XY Coordinates, which add coordinates to points
4. Feature Class To Shapefile. This step dumps points to a shapefile.

### 2.3 Shapefile to CDB
Just use the script shp2cdb.py, which is under "RayJoin/misc/". Example: `python3 shp2cdb.py a.shp b.cdb`

## 3. Evaluate RayJoin

### 3.1 LSI/PIP query
```shell
./query_exec -poly1 dataset1.cdb \
    -poly2 dataset2.cdb
    -mode=rt \
    -xsect_factor=0.5 \ # reserve queue space to store LSI results
    -warmup=5 \ # warmup rounds
    -repeat=5 \ # number of rounds to evaluate. Average time is reported
    -query=lsi/pip
```

### 3.2 Overlay Analysis
```shell
./polyover_exec -poly1 dataset1.cdb \
    -poly2 dataset2.cdb \
    -mode=rt \
    -grid_size=20000 \
    -xsect_factor=0.5
```

## Trouble Shooting Notes

1. Fix weird bug after changing rt code: `rm -rf /var/tmp/OptixCache_${USER}`
2. To enable printf in Optix Kernel: `export OPTIX_FORCE_DEPRECATED_LAUNCHER=1`
