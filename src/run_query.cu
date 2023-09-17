#include <memory>
#include <random>

#include "app/lsi_grid.h"
#include "app/lsi_lbvh.h"
#include "app/lsi_rt.h"
#include "app/pip_grid.h"
#include "app/pip_lbvh.h"
#include "app/pip_rt.h"
#include "map/planar_graph.h"
#include "rt/primitive.h"
#include "run_query.cuh"
#include "tree/primtive.h"
#include "util/array_view.h"
#include "util/helpers.h"
#include "util/timer.h"

namespace rayjoin {

template <typename CONTEXT_T>
void CheckPIPResult(
    CONTEXT_T& ctx, QueryConfig config,
    const thrust::device_vector<typename CONTEXT_T::map_t::point_t>& points,
    const thrust::device_vector<index_t>& eids) {
  auto& stream = ctx.get_stream();
  auto grid = std::make_shared<UniformGrid>(config.grid_size);
  int base_map_id = 0, query_map_id = 1 - base_map_id;
  grid->AddMapToGrid(ctx, base_map_id, config.profile);
  PIPGrid<CONTEXT_T> pip_grid(ctx, grid);

  LOG(INFO) << "Checking point in polygon";
  pip_grid.Query(stream, query_map_id, points);
  stream.Sync();

  pinned_vector<typename CONTEXT_T::map_t::point_t> h_points = points;
  pinned_vector<index_t> closest_eids_ans = pip_grid.get_closest_eids();
  pinned_vector<index_t> closest_eids_res = eids;
  size_t n_diff = 0;
  size_t n_points = h_points.size();

  CHECK_EQ(closest_eids_res.size(), closest_eids_ans.size());

  auto base_map = ctx.get_map(base_map_id);
  auto scaling = ctx.get_scaling();

  for (size_t point_idx = 0; point_idx < n_points; point_idx++) {
    auto closest_eid_ans = closest_eids_ans[point_idx];
    auto closest_eid_res = closest_eids_res[point_idx];
    // different eid does not mean wrong answer because there are two edges
    // having same coordinates but different eid
    if (closest_eid_res != closest_eid_ans) {
      auto not_hit = std::numeric_limits<index_t>::max();
      auto scaled_p = h_points[point_idx];
      bool diff = false;
      std::string ep_ans = "miss";
      std::string ep_res = "miss";
      std::string scaled_ep_ans = "miss";
      std::string scaled_ep_res = "miss";

      if (closest_eid_ans != not_hit) {
        ep_ans = base_map->EndpointsToString(closest_eid_ans, scaling);
        scaled_ep_ans = base_map->ScaledEndpointsToString(closest_eid_ans);
      }

      if (closest_eid_res != not_hit) {
        ep_res = base_map->EndpointsToString(closest_eid_res, scaling);
        scaled_ep_res = base_map->ScaledEndpointsToString(closest_eid_res);
      }

      if (scaled_ep_res != scaled_ep_ans) {
        diff = true;
      }

      if (diff && n_diff < 10) {
        double2 p{scaling.UnscaleX(scaled_p.x), scaling.UnscaleY(scaled_p.y)};
        printf("point %lu (%.8lf, %.8lf) ans %u, res %u %s %s\n", point_idx,
               p.x, p.y, closest_eid_ans, closest_eid_res, ep_ans.c_str(),
               ep_res.c_str());
        printf("scaled point %lu (%ld, %ld) ans %u, res %u %s %s\n", point_idx,
               scaled_p.x, scaled_p.y, closest_eid_ans, closest_eid_res,
               scaled_ep_ans.c_str(), scaled_ep_res.c_str());
      }

      if (diff) {
        n_diff++;
      }
    }
  }
  if (n_diff != 0) {
    LOG(ERROR) << "Map: " << base_map_id << " Total points: " << n_points
               << " n diff: " << n_diff
               << " Error rate: " << (double) n_diff / n_points * 100 << " %";
  } else {
    LOG(INFO) << "Map: " << base_map_id << " passed check";
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
  auto base_pgraph =
      load_from<coord_t>(config.map_path, config.serialize_prefix);
  int base_map_id = 0, query_map_id = 1;

  auto sample = [&](std::shared_ptr<PlanarGraph<coord_t>> pgraph) {
    if (config.sample == "edges") {
      LOG(INFO) << "Sampling edges from map, sample rate: "
                << config.sample_rate << ", seed: " << config.random_seed;
      pgraph =
          sample_edges_from(pgraph, config.sample_rate, config.random_seed);
    } else if (config.sample == "map") {
      LOG(INFO) << "Sampling map, sample rate: " << config.sample_rate
                << ", seed: " << config.random_seed;
      pgraph = sample_map_from(pgraph, config.sample_rate, config.random_seed);
    }
    return pgraph;
  };

  if (config.sample_map_id == 0) {
    base_pgraph = sample(base_pgraph);
  }

  timer_next("Create Context");
  context_t* ctx;
  LSI<context_t>* lsi;

  timer_next("Generate Workloads");

  if (config.query_path.empty()) {
    ctx = new context_t(base_pgraph);
    auto query_map = GenerateLSIQueries(config, *ctx);
    ctx->set_map(query_map_id, query_map);
  } else {
    auto query_pgraph =
        load_from<coord_t>(config.query_path, config.serialize_prefix);
    if (config.sample_map_id == 1) {
      query_pgraph = sample(query_pgraph);
    }
    ctx = new context_t({base_pgraph, query_pgraph});
  }
  Stream& stream = ctx->get_stream();

  timer_next("Create App");
  if (config.mode == "grid") {
    auto grid = std::make_shared<UniformGrid>(config.grid_size);

    auto* lsi_pip = new LSIGrid<context_t>(*ctx, grid);
    QueryConfigGrid query_config;

    query_config.lb = config.lb;
    query_config.grid_size = config.grid_size;
    query_config.profile = config.profile;

    lsi_pip->set_config(query_config);
    lsi = lsi_pip;
  } else if (config.mode == "lbvh") {
    lsi = new LSILBVH<context_t>(*ctx);
  } else if (config.mode == "rt") {
    auto rt_engine = std::make_shared<RTEngine>();
    RTConfig rt_config = get_default_rt_config(config.exec_root);

    rt_engine->Init(rt_config);
    lsi = new LSIRT<context_t>(*ctx, rt_engine);
  } else {
    LOG(FATAL) << "Invalid index type: " << config.mode;
  }

  timer_next("Init");
  auto d_base_map = ctx->get_map(base_map_id)->DeviceObject();
  auto d_query_map = ctx->get_map(query_map_id)->DeviceObject();
  const auto& scaling = ctx->get_scaling();
  size_t queue_cap =
      (d_base_map.get_edges_num() + d_query_map.get_edges_num()) *
      config.xsect_factor;
  LOG(INFO) << "Queue capacity: " << queue_cap;
  lsi->Init(queue_cap);

  if (config.mode == "grid") {
    timer_next("Build Index");
    auto lsi_grid = dynamic_cast<LSIGrid<context_t>*>(lsi)->get_grid();

    lsi_grid->AddMapsToGrid(*ctx, config.profile);
  } else if (config.mode == "rt") {
    auto lsi_rt = dynamic_cast<LSIRT<context_t>*>(lsi);
    thrust::device_vector<OptixAabb> aabbs;
    auto eid_range =
        std::make_shared<thrust::device_vector<thrust::pair<size_t, size_t>>>();
    QueryConfigRT query_config;
    auto rt_engine = lsi_rt->get_rt_engine();
    auto win_size = config.win;
    auto area_enlarge = config.enlarge;

    timer_next("Adaptive Grouping");
    FillPrimitivesGroup(stream, d_base_map, scaling, win_size, area_enlarge,
                        aabbs, *eid_range);
    stream.Sync();

    timer_next("Build Index");
    query_config.profile = config.profile;
    query_config.fau = config.fau;
    query_config.handle =
        rt_engine->BuildAccelCustom(stream, ArrayView<OptixAabb>(aabbs));
    query_config.eid_range = eid_range;

    lsi_rt->set_config(query_config);
  } else if (config.mode == "lbvh") {
    timer_next("Build Index");
    auto lsi_lbvh = dynamic_cast<LSILBVH<context_t>*>(lsi);
    QueryConfigLBVH query_config;
    thrust::device_vector<segment> primitives;
    auto bvh = std::make_shared<lbvh::bvh<float, segment, aabb_getter>>();

    FillPrimitivesLBVH(stream, d_base_map, scaling, primitives);
    stream.Sync();
    bvh->assign(primitives);
    bvh->construct(config.profile);

    query_config.lbvh = bvh;
    query_config.profile = config.profile;
    lsi_lbvh->set_config(query_config);
  }

  timer_next("Warmup");
  for (int i = 0; i < config.warmup; i++) {
    lsi->Query(stream, query_map_id);
  }

  timer_next("Query", config.repeat);
  ArrayView<typename LSI<context_t>::xsect_t> d_xsects;

  for (int i = 0; i < config.repeat; i++) {
    LOG(INFO) << "Iter: " << i;
    lsi->Query(stream, query_map_id);
  }
  d_xsects = lsi->get_xsects();

  LOG(INFO) << "Query: " << config.gen_n
            << " Intersections: " << d_xsects.size()
            << " Selective: " << (double) d_xsects.size() / config.gen_n
            << " Queue Load Factor: " << (double) d_xsects.size() / queue_cap;

  timer_next("Cleanup");

  delete lsi;
  delete ctx;
  timer_end();
}

void RunPIPQuery(const QueryConfig& config) {
  using context_t = Context<coord_t, coefficient_t>;
  using internal_coord_t = typename context_t::internal_coord_t;
  using point_t = typename context_t::map_t::point_t;

  timer_start();
  timer_next("Read map");
  auto base_pgraph =
      load_from<coord_t>(config.map_path, config.serialize_prefix);
  int base_map_id = 0, query_map_id = 1;

  auto sample = [&](std::shared_ptr<PlanarGraph<coord_t>> pgraph) {
    if (config.sample == "edges") {
      LOG(INFO) << "Sampling edges from map, sample rate: "
                << config.sample_rate << ", seed: " << config.random_seed;
      pgraph =
          sample_edges_from(pgraph, config.sample_rate, config.random_seed);
    } else if (config.sample == "map") {
      LOG(INFO) << "Sampling map, sample rate: " << config.sample_rate
                << ", seed: " << config.random_seed;
      pgraph = sample_map_from(pgraph, config.sample_rate, config.random_seed);
    }
    return pgraph;
  };

  if (config.sample_map_id == 0) {
    base_pgraph = sample(base_pgraph);
  }

  timer_next("Create Context");
  context_t* ctx;
  PIP<context_t>* pip;
  thrust::device_vector<point_t> query_points;

  timer_next("Generate Workloads");

  if (config.query_path.empty()) {
    ctx = new context_t(base_pgraph);
    query_points = rayjoin::GeneratePIPQueries(config, *ctx);
  } else {
    auto query_pgraph =
        load_from<coord_t>(config.query_path, config.serialize_prefix);
    if (config.sample_map_id == 1) {
      query_pgraph = sample(query_pgraph);
    }
    ctx = new context_t({base_pgraph, query_pgraph});
    query_points = ctx->get_map(query_map_id)->get_points();
  }

  Stream& stream = ctx->get_stream();

  ArrayView<typename context_t::map_t::point_t> d_query_points(query_points);

  timer_next("Create App");
  if (config.mode == "grid") {
    auto grid = std::make_shared<UniformGrid>(config.grid_size);
    auto* pip_grid = new PIPGrid<context_t>(*ctx, grid);
    QueryConfigGrid config;

    config.grid_size = config.grid_size;
    config.profile = config.profile;

    pip_grid->set_config(config);
    pip = pip_grid;
  } else if (config.mode == "rt") {
    auto rt_engine = std::make_shared<RTEngine>();
    RTConfig rt_config = get_default_rt_config(config.exec_root);

    rt_engine->Init(rt_config);
    pip = new PIPRT<context_t>(*ctx, rt_engine);
  } else if (config.mode == "lbvh") {
    pip = new PIPLBVH<context_t>(*ctx);
  } else {
    LOG(FATAL) << "Invalid index type: " << config.mode;
  }

  timer_next("Init");
  auto d_base_map = ctx->get_map(base_map_id)->DeviceObject();
  const auto& scaling = ctx->get_scaling();
  pip->Init(query_points.size());

  if (config.mode == "grid") {
    timer_next("Build Index");
    auto grid = dynamic_cast<PIPGrid<context_t>*>(pip)->get_grid();

    grid->AddMapToGrid(*ctx, 0, config.profile);
  } else if (config.mode == "rt") {
    auto pip_rt = dynamic_cast<PIPRT<context_t>*>(pip);
    auto rt_engine = pip_rt->get_rt_engine();
    thrust::device_vector<OptixAabb> aabbs;
    auto eid_range =
        std::make_shared<thrust::device_vector<thrust::pair<size_t, size_t>>>();
    auto ne = d_base_map.get_edges_num();

    timer_next("Adaptive Grouping");
    FillPrimitivesGroup(stream, d_base_map, scaling, config.win, config.enlarge,
                        aabbs, *eid_range);
    stream.Sync();

    timer_next("Build Index");
    ArrayView<OptixAabb> d_aabbs(aabbs);

    QueryConfigRT pip_config;

    pip_config.profile = config.profile;
    pip_config.fau = config.fau;
    pip_config.eid_range = eid_range;
    pip_config.handle = rt_engine->BuildAccelCustom(stream, d_aabbs);

    pip_rt->set_config(pip_config);
  } else if (config.mode == "lbvh") {
    timer_next("Build Index");
    auto pip_lbvh = dynamic_cast<PIPLBVH<context_t>*>(pip);
    QueryConfigLBVH query_config;
    thrust::device_vector<segment> primitives;
    auto bvh = std::make_shared<lbvh::bvh<float, segment, aabb_getter>>();

    FillPrimitivesLBVH(stream, d_base_map, scaling, primitives);
    stream.Sync();  // Wait filling primitive finish

    bvh->assign(primitives);
    bvh->construct(config.profile);

    query_config.lbvh = bvh;
    query_config.profile = config.profile;
    pip_lbvh->set_config(query_config);
  }
  stream.Sync();

  timer_next("Warmup");

  for (int i = 0; i < config.warmup; i++) {
    pip->Query(stream, query_map_id, query_points);
  }
  stream.Sync();

  timer_next("Query", config.repeat);
  for (int i = 0; i < config.repeat; i++) {
    pip->Query(stream, query_map_id, query_points);

    if (i == config.repeat - 1) {
      stream.Sync();
      if (config.check && config.mode != "grid") {
        timer_next("Check");
        CheckPIPResult(*ctx, config, query_points, pip->get_closest_eids());
      }
    }
  }

  timer_next("Cleanup");

  delete pip;
  timer_end();
}
}  // namespace rayjoin