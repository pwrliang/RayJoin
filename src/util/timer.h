/**
 * NOLINT(legal/copyright)
 *
 * The file examples/analytical_apps/timer.h is referred and derived from
 * project atlarge-research/graphalytics-platforms-powergraph,
 *
 *    https://github.com/atlarge-research/graphalytics-platforms-powergraph/
 * blob/master/src/main/c/utils.hpp
 *
 * which has the following license:
 *
 *  Copyright 2015 Delft University of Technology
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *          http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef RAYJOIN_UTIL_TIMER_H
#define RAYJOIN_UTIL_TIMER_H

#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

/**
 * Timers for LDBC benchmarking, referred and derived from project
 * atlarge-research/graphalytics-platforms-powergraph.
 */
static double timer() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec / 1000000.0;
}

static bool timer_enabled;
static std::vector<std::tuple<std::string, double, int>> timers;

static void timer_start(bool enabled = true) {
  timers.clear();
  timer_enabled = enabled;
}

static void timer_next(const std::string& name, int repeat = 1) {
  if (timer_enabled) {
    timers.emplace_back(std::make_tuple(name, timer(), repeat));
  }
}

static void timer_end() {
  if (timer_enabled) {
    timer_next("end");

    std::cerr << "Timing results:" << std::endl;

    for (size_t i = 0; i < timers.size() - 1; i++) {
      std::string& name = std::get<0>(timers[i]);
      double time = std::get<1>(timers[i + 1]) - std::get<1>(timers[i]);

      std::cerr << " - " << name << ": " << time * 1000 / std::get<2>(timers[i])
                << " ms" << std::endl;
      std::cerr << std::endl;
    }

    timers.clear();
  }
}
#endif  // RAYJOIN_UTIL_TIMER_H
