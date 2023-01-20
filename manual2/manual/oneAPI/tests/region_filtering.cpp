#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#define CATCH_CONFIG_MAIN
#include "externals/catch2/catch.hpp"

#include <iostream>

#include <random>
#include <algorithm>

#include "oneAPI/quad/Sub_regions.h"
#include "oneAPI/quad/Sub_regions_filter.h"
#include "oneAPI/quad/Region_estimates.h"
#include "oneAPI/quad/util/MemoryUtil.h"
#include "oneAPI/quad/active_regions.h"

double largest_errorest_to_pass(const double estimate, const double epsrel){
    return estimate*epsrel;
}

template<typename T, size_t size>
void set_with_randoms_in_range(std::array<T, size>& arr, quad::Range<double> acceptable_range = {0., 1.}){
    //random generation taken from https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(acceptable_range.low, acceptable_range.high);
    
    for(auto &i : arr)
        i = dis(gen);
}

/*TEST_CASE("Determine number of active sub-regions + errorest filtering")
{
    sycl::queue q;
    const double estimate = 49.13;
    double epsrel = 1.e-3;
    constexpr size_t ndim = 2;
    size_t partitions_per_axis = 50;

    constexpr size_t num_regions = 2500;
    //initialize structures that must be used to inteface with Pagani's methods
    Sub_regions<ndim> sub_regions(q, partitions_per_axis);
    Region_characteristics<ndim> region_characteristics(q, num_regions);
    Region_estimates<ndim> region_estimates(q, num_regions);
    Region_estimates<ndim> parent_estimates(q, num_regions/2); //this doesn't matter for test so we leave uninitialized
    
    //initialize on host the values to copy to the device data-structures above
    std::array<double, num_regions> errors = {0.};
    std::array<double, num_regions> estimates = {0.};
    std::array<double, num_regions> active_flags = {0};
    
    std::fill(estimates.begin(), estimates.end(), estimate);
    
    Sub_regions_filter<ndim> region_filter(q, num_regions);
    
    SECTION("All sub-regions should be finished")
    {
        std::fill(active_flags.begin(), active_flags.end(), 0);
        quad::Range<double> acceptable_range = {0.000001, largest_errorest_to_pass(estimate, epsrel)};
        set_with_randoms_in_range(errors, acceptable_range);
        
        
        memcpy(region_estimates.error_estimates, errors.data(), sizeof(double)*num_regions);
        memcpy(region_estimates.integral_estimates, estimates.data(), sizeof(double)*num_regions);
        memcpy(region_characteristics.active_regions, active_flags.data(), sizeof(double)*num_regions);
        
        //region_filter.get_num_active_regions is invoked from within region_filter.filter but we test both
        size_t num_active_regions = get_num_active_regions<ndim>(q, region_characteristics);
        CHECK(num_active_regions == 0);
        
        num_active_regions = region_filter.filter(q, sub_regions, region_characteristics, region_estimates, parent_estimates);
        CHECK(num_active_regions == 0);
        
    }
    
    SECTION("No sub-regions are finished")
    {
        std::fill(active_flags.begin(), active_flags.end(), 1.);
        quad::Range<double> not_acceptable_range = {largest_errorest_to_pass(estimate, epsrel)+ 10., largest_errorest_to_pass(estimate, epsrel)+11.};
        
        set_with_randoms_in_range(errors, not_acceptable_range);
               
        memcpy(region_estimates.error_estimates, errors.data(), sizeof(double)*num_regions);
        memcpy(region_estimates.integral_estimates, estimates.data(), sizeof(double)*num_regions);
        memcpy(region_characteristics.active_regions, active_flags.data(), sizeof(double)*num_regions);
        
        size_t num_active_regions = get_num_active_regions<ndim>(q, region_characteristics);
        CHECK(num_active_regions == num_regions);

	//for(int i=0; i < 10; ++i)
	//  std::cout<<"region "<< i << " region_estimates.error_estimates:"<<region_estimates.error_estimates[i]<<std::endl;

	//for(int i=0; i < 10; ++i)
	//  std::cout<<"region "<< i << " region_estimates.integral_estimates:"<<region_estimates.integral_estimates[i]<<std::endl;
	
        num_active_regions = region_filter.filter(q, sub_regions, region_characteristics, region_estimates, parent_estimates);
        CHECK(num_active_regions == num_regions);
	
	//for(int i=0; i < 10; ++i)
	//  std::cout<<"region "<< i << " region_estimates.error_estimates:"<<region_estimates.error_estimates[i]<<std::endl;

	//for(int i=0; i < 10; ++i)
	//  std::cout<<"region "<< i << " region_estimates.integral_estimates:"<<region_estimates.integral_estimates[i]<<std::endl;
	
	
	//std::cout<<"largest ok errorest:"<< largest_errorest_to_pass(estimate, epsrel) << std::endl;
	//for(int i=0; i < 10; ++i)
	//  std::cout<<"parent "<< i << " error-estimate:"<<parent_estimates.error_estimates[i]<<std::endl;
	
        CHECK(quad::array_values_larger_than_val<double>(parent_estimates.error_estimates, num_active_regions, largest_errorest_to_pass(estimate, epsrel)));
    }
    
    SECTION("Some sub-regions are finished")
    {
        //can't do std::array<Range<int>, 10> finished_ranges = {{1,6}, {11,18}, ...etc}, HOW TO DO THIS?
        std::array<int, 10> finished_ranges= {1, 6,     11, 18,     46, 1001,   1002, 1003,     1500, 1509};
        
        memcpy(region_estimates.integral_estimates, estimates.data(), sizeof(double)*num_regions);
        quad::set_device_array_range<double>(q, region_characteristics.active_regions, 0, num_regions-1, 1.); //at first make them all active
        quad::set_device_array_range<double>(q, region_estimates.error_estimates, 0, 
            num_regions-1, largest_errorest_to_pass(estimate, epsrel)+10.); //set all error-estimates as active (large error)
        
        size_t true_num_active_regions = num_regions;
        for(size_t i = 0; i < 10; i+=2){
            quad::Range<int> range(finished_ranges[i], finished_ranges[i+1]);
            
            quad::set_device_array_range<double>(q, region_estimates.error_estimates, 
                range.low, range.high, largest_errorest_to_pass(estimate, epsrel)-.0001);
            quad::set_device_array_range<double>(q, region_characteristics.active_regions, range.low, range.high, 0.);
            true_num_active_regions -= range.high - range.low + 1;
            
        }
        
        size_t num_active_regions = get_num_active_regions<ndim>(q, region_characteristics);
        CHECK(num_active_regions == true_num_active_regions);
                
        num_active_regions = region_filter.filter(q, sub_regions, region_characteristics, region_estimates, parent_estimates);
        CHECK(num_active_regions == true_num_active_regions);
        CHECK(quad::array_values_larger_than_val<double>(parent_estimates.error_estimates, 
            num_active_regions, largest_errorest_to_pass(estimate, epsrel)));
    }
    
    SECTION("Only first sub-region is active")
    {
        memcpy(region_estimates.integral_estimates, estimates.data(), sizeof(double)*num_regions);
        quad::set_device_array_range<double>(q, region_characteristics.active_regions, 0, 0, 1.); //make sub-region region active (status = 1)

        //make all other sub-regions region finished/inactive (status = 0)
        quad::set_device_array_range<double>(q, region_characteristics.active_regions, 1, num_regions-1, 0); 
        
        //set those regions to high errorest
        quad::set_device_array_range<double>(q, region_estimates.error_estimates, 0, 0, largest_errorest_to_pass(estimate, epsrel)+10.); 
       
        //set those regions to appropriately low errorest
        quad::set_device_array_range<double>(q, region_estimates.error_estimates, 1, 
            num_regions-1, largest_errorest_to_pass(estimate, epsrel) - .0001); 
        
        size_t true_num_active_regions = 1;
        size_t num_active_regions = get_num_active_regions<ndim>(q, region_characteristics);
        CHECK(num_active_regions == true_num_active_regions);
                
        num_active_regions = region_filter.filter(q, sub_regions, region_characteristics, region_estimates, parent_estimates);
        CHECK(num_active_regions == true_num_active_regions);
        CHECK(quad::array_values_larger_than_val<double>(parent_estimates.error_estimates, 
            num_active_regions, largest_errorest_to_pass(estimate, epsrel)));
    }
    
    SECTION("Only last sub-region is active")
    {
        memcpy(region_estimates.integral_estimates, estimates.data(), sizeof(double)*num_regions);
        
        //make sub-region region active (status = 1)
        quad::set_device_array_range<double>(q, region_characteristics.active_regions, num_regions-1, num_regions-1, 1.); 
        
        //make all other sub-regions region finished/inactive (status = 0)
        quad::set_device_array_range<double>(q, region_characteristics.active_regions, 0, num_regions-2, 0.); 
        
        quad::set_device_array_range<double>(q, region_estimates.error_estimates, 
                num_regions-1, num_regions-1, largest_errorest_to_pass(estimate, epsrel)+10.); //set those regions to high errorest
        
        //set those regions to appropriately low errorest
        quad::set_device_array_range<double>(q, region_estimates.error_estimates, 
            0, num_regions-2, largest_errorest_to_pass(estimate, epsrel) - .0001); 
        
        size_t true_num_active_regions = 1;
       
        size_t num_active_regions = get_num_active_regions<ndim>(q, region_characteristics);
       
        CHECK(num_active_regions == true_num_active_regions);
                
        num_active_regions = region_filter.filter(q, sub_regions, region_characteristics, region_estimates, parent_estimates);
        
        CHECK(num_active_regions == true_num_active_regions);
        
        CHECK(quad::array_values_larger_than_val<double>(parent_estimates.error_estimates, 
            num_active_regions, largest_errorest_to_pass(estimate, epsrel)));
    }
}*/

TEST_CASE("Filtering results")
{
    sycl::queue q;
    printf("in test case\n");
    const double estimate = 49.13;
    double epsrel = 1.e-3;
    constexpr size_t ndim = 3;
    size_t partitions_per_axis = 50;

    size_t num_regions = static_cast<size_t>(pow(partitions_per_axis, ndim));
     
    Sub_regions<ndim> sub_regions(q, partitions_per_axis);
    
    CHECK(sub_regions.size == num_regions);
    printf("Region_estimates &  Region_characteristics\n");
    Region_characteristics<ndim> region_characteristics(q, sub_regions.size);
    Region_estimates<ndim> region_estimates(q, sub_regions.size);
    Region_estimates<ndim> parent_estimates(q, num_regions/2); //this doesn't matter for test so we leave uninitialized
    

    printf("Creating copies\n");
    Region_estimates<ndim> copies(q, num_regions);
    Region_characteristics<ndim> region_characteristics_copies(q, num_regions);
    
    for(size_t reg = 0; reg < num_regions; ++reg){
        region_estimates.integral_estimates[reg] = 1.;
        region_estimates.error_estimates[reg] = .01;
        region_characteristics.sub_dividing_dim[reg] = 1;
        region_characteristics.active_regions[reg] = 1.;
        copies.integral_estimates[reg] = 1.;
        copies.error_estimates[reg] = .01;
        region_characteristics_copies.sub_dividing_dim[reg] = 1;
    }
    
    //initialize on host the values to copy to the device data-structures above
    
    printf("creating filter\n");
    Sub_regions_filter<ndim> region_filter(q, num_regions);
    printf("calling get_num_active_regions\n");           
    size_t num_active_regions = region_filter.get_num_active_regions(q, region_characteristics);
    std::cout<< "num_active_regions:"<<num_active_regions<<std::endl;
    CHECK(num_active_regions == num_regions);
    
    printf("calling filter\n");
    num_active_regions = region_filter.filter(q, sub_regions, region_characteristics, region_estimates, parent_estimates);
    CHECK(num_active_regions == num_regions);
    double total_vol = 0.;
        
    for(size_t reg = 0; reg < num_regions; ++reg){
                CHECK(region_estimates.integral_estimates[reg] == Approx(copies.integral_estimates[reg]));    
                CHECK(region_estimates.error_estimates[reg] == Approx(copies.error_estimates[reg]));    
                CHECK(region_characteristics_copies.sub_dividing_dim[reg] == region_characteristics.sub_dividing_dim[reg]);
    }
        
    for(size_t reg = 0; reg < num_active_regions; ++reg){
        double reg_vol = 1.;
        CHECK(region_characteristics.sub_dividing_dim[reg] <= ndim);
        CHECK(region_characteristics.sub_dividing_dim[reg] >= 0);
                 
        for(size_t dim = 0; dim < ndim; ++dim){
            reg_vol *= sub_regions.dLength[dim * num_active_regions + reg]; 
            CHECK(sub_regions.dLength[dim * num_active_regions + reg] < 1.);
            CHECK(sub_regions.dLength[dim * num_active_regions + reg] > 0.);
            CHECK(sub_regions.dLeftCoord[dim * num_active_regions + reg] < 1.);
            CHECK(sub_regions.dLeftCoord[dim * num_active_regions + reg] >= 0.);
        }
        total_vol += reg_vol;     
    }
         
    CHECK(total_vol == Approx(1.));
            
    
}
