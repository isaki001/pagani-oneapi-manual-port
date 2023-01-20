#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iostream>
#include "Hybrid/PaganiUtils.cuh"
#include "Hybrid/hybrid.cuh"
#include "Hybrid/mem_util.cuh"

class Integrand_2D{
  public:
    Integrand_2D() = default;
    
    double
    operator()(double x, double y){
        return x*y;
    }
};

TEST_CASE("cubature rules get approx. right answer on 2D easy integrand")
{
    size_t partitions_per_axis = 2;
    constexpr size_t ndim = 2;
    Integrand_2D integrand;
    double true_answer = .25;
    Cubature_rules<ndim> cubature_rules;
    Sub_regions<ndim> sub_regions(partitions_per_axis);
    
    SECTION("TDoesn't write partial results (active flags, individual subregion estimates")
    {
        cuhreResult<double> res = cubature_rules.apply_cubature_integration_rules<Integrand_2D>(integrand, sub_regions);
        CHECK(res.estimate == Approx(true_answer));  
    }
    
    SECTION("Traditional interface and meaningful sub-dividing-dimensions")
    {
        Region_estimates<ndim> subregion_estimates(sub_regions.size);
        Region_characteristics<ndim> region_characteristics(sub_regions.size);
        
        Integrand_2D* d_integrand = make_gpu_integrand<Integrand_2D>(integrand);
        cuhreResult<double> res = cubature_rules.apply_cubature_integration_rules<Integrand_2D>(d_integrand, sub_regions, subregion_estimates, region_characteristics);
        CHECK(res.estimate == Approx(true_answer));  
        CHECK(array_values_smaller_than_val<int, size_t>(region_characteristics.sub_dividing_dim, sub_regions.size, ndim));
    }
}


