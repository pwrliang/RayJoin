
#ifndef APP_RT_QUERY_CONFIG_H
#define APP_RT_QUERY_CONFIG_H
namespace rayjoin {

struct RTQueryConfig {
  bool use_triangle;
  bool fau = true;
  float early_term_deviant = 1;
  float epsilon = 0.0001;
};
}  // namespace rayjoin
#endif  // APP_RT_QUERY_CONFIG_H
