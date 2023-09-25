BIN_HOME_RELEASE="/local/storage/liang/.clion/RayJoin/cmake-build-release-dl190"
BIN_HOME_DEBUG="/local/storage/liang/.clion/RayJoin/cmake-build-debug-dl190"
DATASET_ROOT="/local/storage/liang/Downloads/Datasets"
#MAPS=(Aquifers.cdb dtl_cnty.cdb Parks.cdb USACensusBlockGroupBoundaries.cdb USAZIPCodeArea.cdb USADetailedWaterBodies.cdb)
#MAPS1=(Aquifers.cdb Parks.cdb USAZIPCodeArea.cdb)
#MAPS2=(dtl_cnty.cdb USACensusBlockGroupBoundaries.cdb USADetailedWaterBodies.cdb)
MAPS1=(dtl_cnty USACensusBlockGroupBoundaries)
MAPS2=(USAZIPCodeArea USADetailedWaterBodies)
CONTINENTS=(Africa Asia Australia Europe North_America South_America)

SAMPLE_RATES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
DEFAULT_GRID_SIZE=15000
SERIALIZE_PREFIX="/dev/shm"
DEFAULT_XSECT_FACTOR="0.1"
DEFAULT_WIN_SIZE=32
DEFAULT_ENLARGE_LIM=5
DEFAULT_N_WARMUP=5
DEFAULT_N_REPEAT=5