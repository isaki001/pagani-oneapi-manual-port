#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#define CATCH_CONFIG_MAIN
#include "externals/catch2/catch.hpp"
#include <CL/sycl.hpp>
#include "oneAPI/quad/util/cuhreResult.h"
#include "oneAPI/quad/Finished_estimates.h"
#include "oneAPI/quad/Region_estimates.h"
#include "oneAPI/quad/Region_characteristics.h"
#include "oneapi/mkl.hpp"
#include <vector>
#include <array>

using namespace sycl;

TEST_CASE("Manual Dot product")
{
  std::array<double, 5> arr = {1.1, 2.2, 3.3, 4.4, 5.5};
  std::array<double, 5> classif = {0., 1., 0., 1., 1.};
  double true_result = 2.2 + 4.4 + 5.5;
  
  sycl::queue q;
    
  double* _arr = malloc_shared<double>(5, q);
  double* _classif = malloc_shared<double>(5, q);
  double* _res = malloc_shared<double>(5, q);
    
  for(int i = 0; i < 5; ++i){
    _arr[i] = arr[i];
    _classif[i] = classif[i];
  }
    
  double val = 1.3;
  const int stride = 1;
  sycl::event done = oneapi::mkl::blas::row_major::dot(q, 5, _arr, stride, _classif, stride , _res);
  done.wait();
  CHECK(true_result == Approx(_res[0]));
  //std::cout<<"Result:"<< _res[0] << "\n";
}

TEST_CASE("Compute finished estimates")
{
    sycl::queue q;
    constexpr size_t ndim = 2;
    size_t num_regions = 100;
    Region_estimates<ndim> estimates(q, num_regions);
    Region_characteristics<ndim> characteristics(q, num_regions);
    
    double uniform_estimate = 3.2;
    double uniform_errorest =  .00001;
    
    size_t nregions = estimates.size;
    for(int i = 0; i < nregions; ++i){
        estimates.integral_estimates[i] = uniform_estimate;
        estimates.error_estimates[i] = uniform_errorest;
    }
    
    SECTION("All finished regions")
    {
        for(int i = 0; i < nregions; ++i){
            characteristics.active_regions[i] = 0.;
        }
        
        cuhreResult<double> true_iter_estimate;
        true_iter_estimate.estimate = uniform_estimate * static_cast<double>(nregions);
        true_iter_estimate.errorest = uniform_errorest * static_cast<double>(nregions);
        
        cuhreResult<double> test = compute_finished_estimates(q, estimates, characteristics, true_iter_estimate);
        CHECK(true_iter_estimate.estimate == Approx(test.estimate));
        CHECK(true_iter_estimate.errorest == Approx(test.errorest));
    }
    
    SECTION("Few active regions bundled together")
    {
        cuhreResult<double> true_iter_estimate;
        true_iter_estimate.estimate = uniform_estimate * static_cast<double>(nregions);
        true_iter_estimate.errorest = uniform_errorest * static_cast<double>(nregions);
        
        double active_status = 1.;
        size_t first_index = 11;    //first active region
        size_t last_index = 17;     //last active region
        double num_true_active_regions = static_cast<double>(last_index - first_index + 1);
        
        cuhreResult<double> true_iter_finished_estimate;
        true_iter_finished_estimate.estimate = uniform_estimate * static_cast<double>(nregions) - uniform_estimate * num_true_active_regions;
        true_iter_finished_estimate.errorest = uniform_errorest * static_cast<double>(nregions) - uniform_errorest * num_true_active_regions;
        
        for(int i = 0; i < nregions; ++i){
            bool in_active_range = i >= first_index && i <= last_index;
            characteristics.active_regions[i] =  in_active_range ? 1. : 0.;
        }
        
        cuhreResult<double> test = compute_finished_estimates(q, estimates, characteristics, true_iter_estimate);
        CHECK(test.estimate == Approx(true_iter_finished_estimate.estimate));
        CHECK(test.errorest == Approx(true_iter_finished_estimate.errorest));
    }
}
