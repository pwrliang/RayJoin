#ifndef RAYJOIN_POLYGON_PARSER_H
#define RAYJOIN_POLYGON_PARSER_H
#include <glog/logging.h>
#include <unistd.h>

#include <fstream>
#include <sstream>
#include <vector>

#include "util/type_traits.h"
namespace rayjoin {

template <typename ID_T, typename COORD_T, typename OFFSET_T = int>
void ParsePolygons(const char* path, std::vector<ID_T>& vec_idx,
                   std::vector<OFFSET_T>& vec_offsets,
                   std::vector<COORD_T>& vec_x, std::vector<COORD_T>& vec_y,
                   bool head_eq_tail = false, char coordinate_pair_delim = ' ',
                   char coordinate_delim = ',') {
  CHECK(access(path, R_OK) == 0) << "Can not open: " << path;
  std::ifstream if_stream(path);
  std::string line;
  size_t line_no = 0;
  size_t n_polygons = 0;
  OFFSET_T offset = 0;
  static_assert(std::is_same<ID_T, int>::value, "Must be int");

  while (std::getline(if_stream, line)) {
    std::istringstream iss(line);

    if (line_no == 0) {
      iss >> n_polygons;
      vec_idx.reserve(n_polygons);
      vec_offsets.clear();
      vec_offsets.reserve(n_polygons + 1);
      vec_offsets.push_back(0);
    } else {
      std::string item;
      bool first_item = true;
      size_t last_offset = offset;

      while (std::getline(iss, item, coordinate_pair_delim)) {
        if (first_item) {
          int polygon_id = std::stoi(item);

          CHECK(polygon_id >= 0 && polygon_id < n_polygons)
              << "Invalid polygon id: " << polygon_id;

          vec_idx.push_back(polygon_id);
          first_item = false;
        } else {
          auto pos = item.find(coordinate_delim);
          CHECK(pos != std::string::npos) << "Bad line: " << line;
          auto s_x = item.substr(0, pos);
          auto s_y = item.substr(pos + 1);
          COORD_T x, y;

          if (std::is_same<COORD_T, int>::value) {
            x = std::stoi(s_x);
            y = std::stoi(s_y);
          } else if (std::is_same<COORD_T, float>::value) {
            x = std::stof(s_x);
            y = std::stof(s_y);
          } else {
            LOG(FATAL) << "Unimplemented type: " << typeid(COORD_T).name();
          }

          vec_x.push_back(x);
          vec_y.push_back(y);
          offset++;
        }
      }
      if (head_eq_tail && (vec_x[last_offset] != vec_x[vec_x.size() - 1] ||
                           vec_y[last_offset] != vec_y[vec_y.size() - 1])) {
        vec_x.push_back(vec_x[last_offset]);
        vec_y.push_back(vec_y[last_offset]);
        offset++;
      }

      vec_offsets.push_back(offset);
    }
    line_no++;
  }

  if_stream.close();
}
}  // namespace rayjoin

#endif  // RAYJOIN_POLYGON_PARSER_H
