#include <random>

#include "app/lsi_grid.h"
#include "app/lsi_lbvh.h"
#include "app/lsi_rt.h"
#include "app/pip_grid.h"
#include "app/pip_rt.h"
#include "map/planar_graph.h"
#include "run_query.cuh"
#include "util/timer.h"

namespace rayjoin {

template <typename CONTEXT_T>
void CheckPIPResult(
    CONTEXT_T& ctx, QueryConfig config,
    thrust::device_vector<typename CONTEXT_T::map_t::point_t>& points,
    thrust::device_vector<polygon_id_t>& polygon_ids) {
  auto grid = std::make_shared<UniformGrid>(config.grid_size);
  int im = 0;
  grid->AddMapToGrid(ctx, im, config.profiling);
  PIPGrid<CONTEXT_T> pip_grid(ctx, grid);

  LOG(INFO) << "Checking point in polygon";

  pinned_vector<polygon_id_t> point_in_polygon = pip_grid.Query(im, points);
  pinned_vector<polygon_id_t> point_in_polygon_rt = polygon_ids;
  size_t n_diff = 0;

  CHECK_EQ(point_in_polygon.size(), point_in_polygon_rt.size());

  for (size_t point_idx = 0; point_idx < point_in_polygon.size(); point_idx++) {
    if (point_in_polygon[point_idx] != point_in_polygon_rt[point_idx]) {
      n_diff++;
      if (n_diff < 10) {
        auto p = ctx.get_planar_graph(im)->points[point_idx];

        printf("Diff! Map: %d point: %zu, (%lf, %lf), ans: %d, res: %d\n", im,
               point_idx, (double) p.x, (double) p.y,
               point_in_polygon[point_idx], point_in_polygon_rt[point_idx]);
      }
    }
  }
  if (n_diff != 0) {
    LOG(ERROR) << "Map: " << im << " Total points: " << point_in_polygon.size()
               << " n diff: " << n_diff << " Error rate: "
               << (double) n_diff / point_in_polygon.size() * 100 << " %";
  } else {
    LOG(INFO) << "Map: " << im << " passed check";
  }
}

template <typename CONTEXT_T>
std::shared_ptr<typename CONTEXT_T::map_t> GenerateLSIQueries(
    const QueryConfig& config, CONTEXT_T& ctx) {
  using coord_t = typename CONTEXT_T::coord_t;
  using coefficient_t = typename CONTEXT_T::coefficient_t;
  using point_t = typename cuda_vec<coord_t>::type_2d;
  using edge_t = dev::Edge<coefficient_t>;
  auto bb = ctx.get_bounding_box();
  auto& stream = ctx.get_stream();
  auto scaling = ctx.get_scaling();
  size_t ne = config.gen_n;
  auto max_t = config.gen_t;
  auto seed = config.random_seed;
  std::random_device rd;  // obtain a random number from hardware
  std::mt19937 gen(seed == 0 ? rd() : seed);  // seed the generator
  std::uniform_real_distribution<> dist_x(bb.min_x, bb.max_x);  // unscaled x,y
  std::uniform_real_distribution<> dist_y(bb.min_y, bb.max_y);
  std::uniform_real_distribution<> dist_t(0, max_t);

  auto query_map = std::make_shared<typename CONTEXT_T::map_t>(1);

  pinned_vector<point_t> points(ne * 2);
  pinned_vector<edge_t> edges(ne);

  for (size_t i = 0; i < ne; i++) {
    auto x1 = dist_x(gen), y1 = dist_y(gen);
    auto x2 = dist_x(gen), y2 = dist_y(gen);
    auto len = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    auto d_x = (x2 - x1) / len, d_y = (y2 - y1) / len;
    auto t = dist_t(gen);

    point_t p1{x1, y1};
    point_t p2{x1 + t * d_x, y1 + t * d_y};

    points[2 * i] = p1;
    points[2 * i + 1] = p2;
    edges[i].eid = i;
    edges[i].p1_idx = 2 * i;
    edges[i].p2_idx = 2 * i + 1;
    edges[i].left_polygon_id = edges[i].right_polygon_id = 0;
  }
  query_map->LoadFrom(stream, scaling, points, edges);
  return query_map;
}

template <typename CONTEXT_T>
thrust::host_vector<typename CONTEXT_T::map_t::point_t> GeneratePIPQueries(
    const QueryConfig& config, CONTEXT_T& ctx) {
  using coefficient_t = typename CONTEXT_T::coefficient_t;
  auto bb = ctx.get_bounding_box();
  auto scaling = ctx.get_scaling();
  auto seed = config.random_seed;
  std::random_device rd;  // obtain a random number from hardware
  std::mt19937 gen(seed == 0 ? rd() : seed);  // seed the generator
  std::uniform_real_distribution<> dist_x(bb.min_x, bb.max_x);  // unscaled x,y
  std::uniform_real_distribution<> dist_y(bb.min_y, bb.max_y);

  thrust::host_vector<typename CONTEXT_T::map_t::point_t> points;

  for (size_t i = 0; i < config.gen_n; i++) {
    auto x = dist_x(gen), y = dist_y(gen);

    points.push_back({scaling.ScaleX(x), scaling.ScaleY(y)});
  }

  return points;
}

void RunLSIQuery(const QueryConfig& config) {
  using context_t = Context<coord_t, coefficient_t>;
  using internal_coord_t = typename context_t::internal_coord_t;

  timer_start();
  timer_next("Read map");
  auto base_map = load_from<coord_t>(config.map_path, config.serialize_prefix);

  if (config.sample == "edges") {
    LOG(INFO) << "Sampling edges from map, sample rate: " << config.sample_rate
              << ", seed: " << config.random_seed;
    base_map =
        sample_edges_from(base_map, config.sample_rate, config.random_seed);
  } else if (config.sample == "map") {
    LOG(INFO) << "Sampling map, sample rate: " << config.sample_rate
              << ", seed: " << config.random_seed;
    base_map =
        sample_map_from(base_map, config.sample_rate, config.random_seed);
  }

  timer_next("Create Context");
  context_t ctx(base_map);
  LSI<context_t>* lsi;

  timer_next("Generate Workloads");
  auto query_map = GenerateLSIQueries(config, ctx);
  ctx.set_query_map(query_map);

  timer_next("Create App");
  if (config.mode == "grid") {
    auto grid = std::make_shared<UniformGrid>(config.grid_size);

    lsi = new LSIGrid<context_t>(ctx, grid);
    dynamic_cast<LSIGrid<context_t>*>(lsi)->set_load_balancing(config.lb);
  } else if (config.mode == "lbvh") {
    lsi = new LSILBVH<context_t>(ctx, config.profiling);
  } else if (config.mode == "rt") {
    auto rt_engine = std::make_shared<RTEngine>();
    RTConfig rt_config = get_default_rt_config(config.exec_root);
    RTQueryConfig query_config;

    query_config.use_triangle = config.use_triangle;
    query_config.fau = config.fau;
    query_config.epsilon = config.epsilon;
    query_config.early_term_deviant = config.early_term_deviant;

    rt_engine->Init(rt_config);
    lsi = new LSIRT<context_t>(ctx, rt_engine, query_config);
  } else {
    LOG(FATAL) << "Invalid index type: " << config.mode;
  }

  timer_next("Init");
  size_t queue_cap =
      (ctx.get_map(0)->get_edges_num() + config.gen_n) * config.xsect_factor;
  LOG(INFO) << "Queue capacity: " << queue_cap;
  lsi->Init(queue_cap);

  int query_map_id = 1;

  timer_next("Build Index");
  if (config.mode == "grid") {
    auto grid = dynamic_cast<LSIGrid<context_t>*>(lsi)->get_grid();

    grid->AddMapsToGrid(ctx, config.profiling);
  } else if (config.mode == "rt") {
    dynamic_cast<LSIRT<context_t>*>(lsi)->BuildIndex(query_map_id);
  }

  timer_next("Warmup");
  for (int i = 0; i < config.warmup; i++) {
    lsi->Query(query_map_id);
  }

  timer_next("Query", config.repeat);
  ArrayView<typename LSI<context_t>::xsect_t> d_xsects;

  for (int i = 0; i < config.repeat; i++) {
    d_xsects = lsi->Query(query_map_id);
  }

  LOG(INFO) << "Query: " << config.gen_n
            << " Intersections: " << d_xsects.size()
            << " Selective: " << (double) d_xsects.size() / config.gen_n
            << " Queue Load Factor: " << (double) d_xsects.size() / queue_cap;

  timer_next("Cleanup");

  delete lsi;
  timer_end();
}

void RunPIPQuery(const QueryConfig& config) {
  using context_t = Context<coord_t, coefficient_t>;
  using internal_coord_t = typename context_t::internal_coord_t;

  timer_start();
  timer_next("Read map");
  auto base_map = load_from<coord_t>(config.map_path, config.serialize_prefix);

  timer_next("Create Context");
  context_t ctx(base_map);
  PIP<context_t>* pip;

  timer_next("Generate Workloads");
  thrust::device_vector<typename context_t::map_t::point_t> query_points =
      rayjoin::GeneratePIPQueries(config, ctx);

  timer_next("Create App");
  if (config.mode == "grid") {
    auto grid = std::make_shared<UniformGrid>(config.grid_size);

    pip = new PIPGrid<context_t>(ctx, grid);
  } else if (config.mode == "rt") {
    auto rt_engine = std::make_shared<RTEngine>();
    RTConfig rt_config = get_default_rt_config(config.exec_root);
    RTQueryConfig pip_config;

    pip_config.use_triangle = config.use_triangle;
    pip_config.fau = config.fau;
    pip_config.epsilon = config.epsilon;
    pip_config.early_term_deviant = config.early_term_deviant;

    rt_engine->Init(rt_config);
    pip = new PIPRT<context_t>(ctx, rt_engine, pip_config);
  } else {
    LOG(FATAL) << "Invalid index type: " << config.mode;
  }

  timer_next("Init");
  pip->Init(query_points.size());

  timer_next("Build Index");
  if (config.mode == "grid") {
    auto grid = dynamic_cast<PIPGrid<context_t>*>(pip)->get_grid();

    grid->AddMapToGrid(ctx, 0, config.profiling);
  } else if (config.mode == "rt") {
    dynamic_cast<PIPRT<context_t>*>(pip)->BuildIndex(0);
  }

  timer_next("Warmup");
  ArrayView<typename context_t::map_t::point_t> d_query_points(query_points);

  for (int i = 0; i < config.warmup; i++) {
    pip->Query(0, query_points);
  }

  timer_next("Query", config.repeat);
  for (int i = 0; i < config.repeat; i++) {
    auto& polygon_ids = pip->Query(0, query_points);

    if (i == config.repeat - 1) {
      if (config.check && config.mode == "rt") {
        timer_next("Check");
        CheckPIPResult(ctx, config, query_points, polygon_ids);
      }
    }
  }

  timer_next("Cleanup");

  delete pip;
  timer_end();
}
}  // namespace rayjoin