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
- [USCounty](https://www.arcgis.com/home/item.html?id=14c5450526a8430298b2fa74da12c2f4)
- [Zipcode](https://www.arcgis.com/home/item.html?id=d6f7ee6129e241cc9b6f75978e47128b)
- [BlockGroup](https://www.arcgis.com/home/item.html?id=1c924a53319a491ab43d5cb1d55d8561)
- [Aquifers](https://www.arcgis.com/home/item.html?id=d2ce672fda1f44089af659b629d11458)
- [Parks](https://www.arcgis.com/home/item.html?id=f092c20803a047cba81fbf1e30eff0b5)
- [WaterBodies](https://www.arcgis.com/home/item.html?id=48c77cbde9a0470fb371f8c8a8a7421a)

### 2.2 Process dataset with ArcGIS
We basically need two 4 steps to process the datasets, which can be found in the toolbox in ArcGIS Pro. 
1. Polygon To Line. Make sure "Identify and store polygon neighboring information" is checked.
2. Feature Vertices To Points, which converts the output from step 1 to points.
3. Add XY Coordinates, which add coordinates to points
4. Feature Class To Shapefile. This step dumps points to a shapefile.

### 2.3 Shapefile to CDB
Just use the script shp2cdb.py, which is under "RayJoin/misc/". Example: `python3 shp2cdb.py a.shp b.cdb`

## 3. Evaluate RayJoin
We provide many scripts under "Project/expr" that reproduces the experiment results. You can also manually
evalute RayJoin with the following commands.

### 3.1 LSI
```shell
./query_exec -poly1 dataset.cdb \
    -mode=grid/rt \
    -gen_n=100000 \ # number of line segments
    -gen_t=0.1 \ # max length of line segments
    -xsect_factor=0.5 \ # reserve queue space to store LSI results
    -warmup=5 \ # warmup rounds
    -repeat=5 \ # number of rounds to evaluate. Average time is reported
    -query=lsi
```

### 3.2 PIP
```shell
./query_exec -poly1 dataset.cdb \
    -mode=grid/rt \
    -gen_n=1000000 \ # number of points
    -warmup=5 \ # warmup rounds
    -repeat=5 \ # number of rounds to evaluate. Average time is reported
    -query=pip
```

### 3.3 Overlay Analysis
```shell
./polyover_exec -poly1 dataset1.cdb \
    -poly2 dataset2.cdb \
    -serialize=/dev/shm \ # Serialize CDB to binary format, which saves loading time.
    -mode=grid/rt \
    -grid_size=20000 \
    -epsilon=0.00001 \ # the paramter of Reservative Representation
    -check=true \ # Check Error Rate
    -fau \ # Free AABBs after builing the BVH, which allows loading big datasets at the overhead of free memory 
    -xsect_factor=0.5

```

## 4. Implementation References
### 4.1 [OVERPROP](https://wrfranklin.org/pmwiki/pmwiki.php/Research/OverlayingTwoMaps)
### 4.2 [LBVH](https://github.com/ToruNiina/lbvh)


## Trouble Shooting

1. Fix weird bug after changing rt code: `rm -rf /var/tmp/OptixCache_${USER}`
2. To enable printf in Optix Kernel: `export OPTIX_FORCE_DEPRECATED_LAUNCHER=1`

## Notes from ArcGIS

### [Polygon To Line](https://desktop.arcgis.com/en/arcmap/latest/tools/data-management-toolbox/polygon-to-line.htm)
- In a polygon geometry, the outer boundary is always stored in a clockwise direction. If the polygon has a hole, the hole (or inner) boundary is always stored in a counterclockwise direction. Therefore, for a polygon with no neighbors to the left side (outside) of its outer boundary and the left side (inside) of the hole boundary, the resulting lines will have a value of -1 for LEFT_FID and the polygon feature ID as the RIGHT_FID.
- If a polygon contains another polygon, one output line in the clockwise direction will be generated representing the shared boundary, with its LEFT_FID set to the outer polygon feature ID and the RIGHT_FID set to the inner polygon feature ID.
- If two polygons share a portion of their boundaries, one output line will be generated representing the shared segment. The line direction will be arbitrary; the LEFT_FID and the RIGHT_FID will be set to the left or right polygon feature IDs accordingly.
- If a polygon overlaps another polygon, two output lines will be generated representing each crossing boundary twice: the first line will represent the outer boundary of one of the overlapping polygons, therefore, its LEFT_FID is the feature ID of the polygon it crosses, and its RIGHT_FID will be its own polygon feature ID; the second line will be in the opposite direction, splitting the other polygon, therefore, its LEFT_FID and RIGHT_FID will be the same as the other polygon feature ID.
- Multiparts in input polygons are not maintained; the output lines are all single part.


### Reproduce
### Scalability
1. LSI `./run_scalability.sh -s=map --lsi-query`
2. PIP `./run_scalability.sh -s=points --pip-query`
### AG
1. LSI `./run_ag.sh --lsi-vary-enlarge-lim`
2. PIP `./run_ag.sh --pip-vary-enlarge-lim`
### Evaluate Grid,LBVH,RayJoin for LSI+PIP+Overlay queries
1. `./run_e2e.sh -ov`