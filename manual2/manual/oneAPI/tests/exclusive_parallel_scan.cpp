#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#define CATCH_CONFIG_MAIN
//#include "externals/catch2/catch.hpp"
#include "externals/catch2/catch.hpp"
#include <CL/sycl.hpp>

#include "oneAPI/quad/Region_characteristics.h"
#include "oneAPI/quad/Finished_estimates.h"

//#include "oneAPI/quad/active_regions.h"
#include "oneAPI/quad/Sub_regions_filter.h"

TEST_CASE("All finished regions")
{
  sycl::queue q;
  const size_t num_regions = 10;
  Region_characteristics<2> regs(q, num_regions);  
  Sub_regions_filter<2> filter(q, num_regions);
  
  for(int i=0; i < num_regions; ++i)
      regs.active_regions[i] = 0.;  

  size_t num_active = filter.get_num_active_regions(q, regs);    
  CHECK(num_active == 0);  
}

TEST_CASE("No finished regions")
{
  sycl::queue q;
  const size_t num_regions = 10;
  Region_characteristics<2> regs(q, num_regions);  
  Sub_regions_filter<2> filter(q, num_regions);
  
  for(int i=0; i < num_regions; ++i)
      regs.active_regions[i] = 1.;  

  size_t num_active = filter.get_num_active_regions(q, regs);    
  CHECK(num_active == 10);
}


TEST_CASE("Some sub-regions are finished")
{
  const size_t num_ranges_with_all_finished = 5;
  std::array<std::pair<size_t, size_t>, num_ranges_with_all_finished> finished_ranges = {{{0, 50},{63, 71},{ 101, 121},{124, 125},{127, 129}}};
 
  sycl::queue q;
  const size_t num_regions = 1000;
  Region_characteristics<2> regs(q, num_regions);  
  Sub_regions_filter<2> filter(q, num_regions);  
    
  for(int i=0; i < num_regions; ++i)
      regs.active_regions[i] = 1.;  
  
  for(auto range : finished_ranges){
      //set finished regions from ranges
      for(int i=range.first; i <= range.second; ++i)
          regs.active_regions[i] = 0.;       
  }  
    
    
  size_t num_true_active = 1000;
  for(int i = 0; i < num_ranges_with_all_finished; ++i){
      num_true_active -= finished_ranges[i].second - finished_ranges[i].first + 1;
  }  
    
  size_t num_active = filter.get_num_active_regions(q, regs);    
  CHECK(num_active == num_true_active);
}

TEST_CASE("Only first and last are finished")
{ 
  sycl::queue q;
  const size_t num_regions = 1000;
  Region_characteristics<2> regs(q, num_regions);  
  Sub_regions_filter<2> filter(q, num_regions);
  
  for(int i=0; i < num_regions; ++i)
      regs.active_regions[i] = 1.;  
    
  regs.active_regions[0] = 0.;  
  regs.active_regions[num_regions-1] = 0.;  
    
  size_t num_active = filter.get_num_active_regions(q, regs);          
  CHECK(num_active == 998);
}
