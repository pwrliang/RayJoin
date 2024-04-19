# **RayJoin**

RayJoin utilizes the ray tracing hardware in modern GPUs (e.g., NVIDIA RT Cores) 
as accelerators to achieve high-performance spatial join processing.
Specifically, RayJoin consists of a high-performance and high-precision spatial join framework 
that accelerates two vital spatial join queries: line segment intersection (LSI) and point-in-polygon test (PIP). 
Polygon overlay analysis is also supported by combining the query results of LSI and PIP. Besides these ray tracing-backed
algorithms, RayJoin also contains new solutions to address two challenging technical issues: (1) how to meet the high precision
requirement of spatial data analysis with the insufficient precision support by the underlying hardware, and (2) how to reduce the high
buildup cost of the hardware-accelerated index, namely Bounding Volume Hierarchy (BVH), while maintaining optimal query exe-
cution times. RayJoin achieves speedups from 3.0x to 28.3x over any existing highly optimized methods in high precision. To the best of our knowledge, RayJoin
stands as the sole solution capable of meeting the real-time requirements of diverse workloads, taking under 460ms to join millions of polygons.

## Build

### Install Dependencies

(1) [gflags](https://github.com/gflags/gflags)

(2) [glog](https://github.com/google/glog)

(3) [NVIDIA Optix 8.0](https://developer.nvidia.com/designworks/optix/download)

(4) [NVIDIA CUDA 12.3+](https://developer.nvidia.com/cuda-11-6-0-download-archive)

(5) [NVIDIA Driver 530+](https://www.nvidia.com/download/index.aspx)

Install gflags and glog:
`sudo apt install libgflags-dev libgoogle-glog-dev`

Install Optix: 
```shell
export OPTIX_HOME=~/optix

mkdir -p $OPTIX_HOME
./NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh --prefix=$OPTIX_HOME --exclude-subdir --skip-license
```

### Building Instructions

- Debug (Building the project under the debug mode enables some counter to profile RayJoin)
```shell
mkdir debug
cd debug
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=$OPTIX_HOME ..
make
```

- Release
```shell
mkdir release
cd release
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$OPTIX_HOME ..
make
```

After the project is successfully built, two binary called `polyover_exec` and `query_exec` will be generated under the `bin` of the building path.
## Dataset Preparation

### Download Datasets

We do not provide preprocessed datasets for now, because uploading datasets will expose our identification. 
You need to download and process by yourself. 
A very small sample dataset is included under the `test` folder, which allows you to try out RayJoin.

- [USCounty](https://www.arcgis.com/home/item.html?id=14c5450526a8430298b2fa74da12c2f4)
- [Zipcode](https://www.arcgis.com/home/item.html?id=d6f7ee6129e241cc9b6f75978e47128b)
- [BlockGroup](https://www.arcgis.com/home/item.html?id=1c924a53319a491ab43d5cb1d55d8561)
- [WaterBodies](https://www.arcgis.com/home/item.html?id=48c77cbde9a0470fb371f8c8a8a7421a)
- [Lakes and Parks](https://spatialhadoop.cs.umn.edu/datasets.html)


### Datasets format

RayJoin requires CDB format to work, which is described in this [paper](https://dl.acm.org/doi/abs/10.1145/2835185.2835188). We may support more formats in the future, but for now, only CDB format is supported.
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

We provide [preprocessed datasets](https://datadryad.org/stash/share/aIs0nLs2TsLE_dcWO2qPHiohRKoOI3kx0WGT5BnATtA) . You can also generate CDB datasets with the following steps:

1. If you use the datasets from ArcGIS hub, you download and save them into shapefile. If the datasets are from SpatialHadoop website, they are in the Well-known Text (WKT) format, and you need to convert them into shapefile with `misc/wkt2shp.py input.wkt output.shp`.
2. Load the shapefile in ArcGIS, and go through the two following steps. We need to convert polygons in shapefile into polylines with neighboring information.
- Polygon To Line. Make sure "Identify and store polygon neighboring information" is checked.
- Feature Class To Shapefile. This step dumps polylines to a shapefile.
3. Run the script `misc/shp2cdb.py input.shp output.cdb`




## Evaluate **RayJoin**

### Parameters

- `-poly1` path of the base map (*R*) in CDB format
- `-poly2` path of the query map (*S*) in CDB format
- `-mode` implementation used to run the query, including `grid`, `lbvh`, `rt`
- `-xsect_factor` reserve a queue to store LSI results. The queue capacity is calculated by `|R|*|S|*xsect_factor`
- `-query` query type, which can be `lsi` or `pip`. This parameter only works for `query_exec`
- `-check` compare results with the grid implementation. Only works when the mode is `rt`
- `-output` output path of polygon overlay results. Only works for `polyover_exec`

### Run LSI and PIP Queries

```shell
./query_exec -poly1 base_map.cdb \
    -poly2 query_map.cdb \
    -mode=grid/lbvh/rt \
    -xsect_factor=0.1 \ 
    -warmup=5 \ # warmup rounds
    -repeat=5 \ # number of rounds to evaluate. Average time is reported
    -query=lsi/pip
```

### Run Overlay Analysis

```shell
./polyover_exec -poly1 dataset1.cdb \
    -poly2 dataset2.cdb \
    -serialize=/dev/shm \ # Serialize CDB to binary format, which saves loading time.
    -mode=grid/rt \
    -check=true \ # Compare results with the uniform grid implementation
    -xsect_factor=0.5
```

### Examples

We provided a sample dataset and test script under `test`, which allows you to try out RayJoin without figuring out what these parameters work for. 
Be sure you have built the project in debug and release mode before run the script.


## Trouble Shooting

1. Fix weird bug after changing rt code: `rm -rf /var/tmp/OptixCache_${USER}`
2. To enable printf in Optix Kernel: `export OPTIX_FORCE_DEPRECATED_LAUNCHER=1`

## References

1. [OVERPROP](https://wrfranklin.org/pmwiki/pmwiki.php/Research/OverlayingTwoMaps) helps us a lot for the polygon overlay implementation.
2. We use [this library](https://github.com/ToruNiina/lbvh) as the LBVH implementation. 