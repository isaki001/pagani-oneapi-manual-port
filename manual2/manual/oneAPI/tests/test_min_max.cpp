#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#include "oneAPI/quad/heuristic_classifier.h"

#define CATCH_CONFIG_MAIN
#include "externals/catch2/catch.hpp"


#include <vector>
#include <array>
#include <iostream>

using namespace sycl;

TEST_CASE("using buffer")
{
  sycl::queue q;  
        
  double* estimates = sycl::malloc_shared<double>(1000, q);  
  double* min = sycl::malloc_shared<double>(1, q);  
  double* max = sycl::malloc_shared<double>(1, q);

  for(int i=0; i < 1000; ++i)
    estimates[i] = static_cast<double>(i);
  
  auto range = device_array_min_max<double>(q, estimates, 1000);
  
  
  CHECK(range.low == 0.);
  CHECK(range.high == 999.);
}
