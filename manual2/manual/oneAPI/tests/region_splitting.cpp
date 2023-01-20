#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#define CATCH_CONFIG_MAIN
#include "externals/catch2/catch.hpp"     
#include <iostream>

#include "oneAPI/quad/Sub_region_splitter.h"

#include <random>
#include <algorithm>

template<size_t ndim>
bool is_free_of_duplicates(Sub_regions<ndim>& regions){
    for(size_t regionID = 0; regionID < regions.size; regionID++){
        quad::Volume<double, ndim> region = regions.extract_region(regionID);
        for(size_t reg = 0; reg < regions.size; reg++){
            quad::Volume<double, ndim> region_i = regions.extract_region(reg);
            if(reg != regionID && region == region_i){
                return false;
            }
        }
    }
    return true;
}

double largest_errorest_to_pass(const double estimate, const double epsrel){
    return estimate*epsrel;
}

void
write_axis_lengths(double* length, double* output, size_t num_regions, size_t dim){
    for(int i=0; i < num_regions; ++i){
        output[i] = length[dim*num_regions + i];
    }
}

template<size_t num_regions, size_t ndim>
std::array<double, num_regions> check_sub_region_lengths(sycl::queue& q, const Sub_regions<ndim>& sub_regions, size_t dim_to_check){
    
  double* axis_lengths = sycl::malloc_shared<double>(num_regions, q);
  std::array<double, num_regions> axis_lengths_at_dim;
  write_axis_lengths(sub_regions.dLength, axis_lengths, num_regions, dim_to_check);
    
    
  memcpy(axis_lengths_at_dim.begin(), axis_lengths, sizeof(double)* num_regions);
  return axis_lengths_at_dim;
    
}

template<size_t ndim>
double
sub_region_length_at_dim(sycl::queue& q, const Sub_regions<ndim>& regions, size_t region_id, size_t dim_to_check){
  double* length = sycl::malloc_shared<double>(1, q);
  size_t index = dim_to_check * regions.size + region_id;
  memcpy(length, regions.dLength + index, sizeof(double));
  const double return_val = *length;
  free(length, q);
  return return_val;                       
}


void set_with_random_split_dim(int* arr, size_t size, int ndim){
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_int_distribution<int> distr(0, ndim-1);
    
    for(int i=0; i< size; ++i){
        arr[i] = distr(eng);
    }
}

constexpr double const_pow(double x, double y){
    double res = x;
    for(size_t i = 1; i < y; ++i)
        res *= x; 
    return res;
}

constexpr 
size_t
compute_num_regions(size_t ndim, size_t sub_regions_per_axis){
    double res = const_pow(static_cast<double>(sub_regions_per_axis), static_cast<double>(ndim));
    return static_cast<size_t>(res);
}

TEST_CASE("2D Space splits"){
  sycl::queue q;
  constexpr size_t ndim = 2;
  const size_t sub_regions_per_axis = 10;
  const size_t num_regions = compute_num_regions(ndim, sub_regions_per_axis);//100;
  CHECK(num_regions == 100);
    
  Sub_regions<ndim> sub_regions(q, sub_regions_per_axis);
  Sub_region_splitter<ndim> splitter(q, sub_regions.size);
  CHECK(num_regions == sub_regions.size);
    
  Region_characteristics<ndim> classifiers(q, sub_regions.size);
  std::array<int, num_regions> sub_dividing_dim;
  set_with_random_split_dim(sub_dividing_dim.data(), num_regions, ndim);
  memcpy(classifiers.sub_dividing_dim, sub_dividing_dim.data(), sizeof(int) * sub_regions.size);
    
  splitter.split(q, sub_regions, classifiers);
    
  CHECK(num_regions * 2 == sub_regions.size);
  CHECK(is_free_of_duplicates<ndim>(sub_regions));
  CHECK(Approx(1.0).epsilon(0.00001) == sub_regions.compute_total_volume());
}

TEST_CASE("3D Space splits"){
  sycl::queue q;
  constexpr size_t ndim = 3;
  constexpr size_t sub_regions_per_axis = 5;
  constexpr size_t num_regions = compute_num_regions(ndim, sub_regions_per_axis);
    
  Sub_regions<ndim> sub_regions(q, sub_regions_per_axis);
  Sub_region_splitter<ndim> splitter(q, sub_regions.size);
  CHECK(num_regions == sub_regions.size);
    
  Region_characteristics<ndim> classifiers(q, sub_regions.size);
  std::array<int, num_regions> sub_dividing_dim;
  set_with_random_split_dim(sub_dividing_dim.data(), num_regions, ndim);
  memcpy(classifiers.sub_dividing_dim, sub_dividing_dim.data(), sizeof(int) * sub_regions.size);
    
  splitter.split(q, sub_regions, classifiers);
    
  CHECK(num_regions * 2 == sub_regions.size);
  CHECK(is_free_of_duplicates<ndim>(sub_regions));
  CHECK(Approx(1.0).epsilon(0.00001) == sub_regions.compute_total_volume());
}

TEST_CASE("8D Space splits"){
  sycl::queue q;
  constexpr size_t ndim = 8;
  constexpr size_t sub_regions_per_axis = 5;
  constexpr size_t num_regions = compute_num_regions(ndim, sub_regions_per_axis);
    
  Sub_regions<ndim> sub_regions(q, sub_regions_per_axis);
  Sub_region_splitter<ndim> splitter(q, sub_regions.size);
  CHECK(num_regions == sub_regions.size);
    
  Region_characteristics<ndim> classifiers(q, sub_regions.size);
  std::array<int, num_regions> sub_dividing_dim;
  set_with_random_split_dim(sub_dividing_dim.data(), num_regions, ndim);
  memcpy(classifiers.sub_dividing_dim, sub_dividing_dim.data(),
	     sizeof(int) * sub_regions.size);
    
  splitter.split(q, sub_regions, classifiers);
    
  CHECK(num_regions * 2 == sub_regions.size);
  //CHECK(is_free_of_duplicates<ndim>(&sub_regions)); //takes too long, consider gpu implementation
  CHECK(Approx(1.0).epsilon(0.00001) == sub_regions.compute_total_volume());
}

TEST_CASE("Check sub-region lenghts after split")
{
  sycl::queue q;
  constexpr size_t ndim = 2;
  const size_t sub_regions_per_axis = 50;
  constexpr size_t pre_split_num_regions = 2500;
    
  constexpr size_t post_split_num_regions = 5000;
  const double length_pre_split = 1./50.;
  const double length_post_split = length_pre_split/2.;
    
  Sub_regions<ndim> sub_regions(q, sub_regions_per_axis);
  Region_characteristics<ndim> classifiers(q, pre_split_num_regions);
  std::array<int, pre_split_num_regions> sub_dividing_dim;
    
  Sub_region_splitter<ndim> splitter(q, pre_split_num_regions);
    
  SECTION("Split dim 0 on all regions"){
    std::fill(sub_dividing_dim.begin(), sub_dividing_dim.end(), 0);
    memcpy(classifiers.sub_dividing_dim, sub_dividing_dim.data(), sizeof(int) * pre_split_num_regions);
                           
    splitter.split(q, sub_regions, classifiers);
    size_t dim_to_check = 0;
    std::array<double, post_split_num_regions> axis_lengths_at_dim = check_sub_region_lengths<post_split_num_regions, ndim>(q, sub_regions, dim_to_check);
    for(auto i: axis_lengths_at_dim)
      CHECK(i == Approx(length_post_split));
        
    dim_to_check = 1;
    axis_lengths_at_dim = check_sub_region_lengths<post_split_num_regions, ndim>(q, sub_regions, dim_to_check);
    for(auto i: axis_lengths_at_dim)
      CHECK(i == Approx(length_pre_split));
  }
    
  SECTION("Split dim 1 on all regions"){
    std::fill(sub_dividing_dim.begin(), sub_dividing_dim.end(), 1);
    memcpy(classifiers.sub_dividing_dim, sub_dividing_dim.data(),
	       sizeof(int) * pre_split_num_regions);
    splitter.split(q, sub_regions, classifiers);     
        
    size_t dim_to_check = 0;
    std::array<double, post_split_num_regions> axis_lengths_at_dim = check_sub_region_lengths<post_split_num_regions, ndim>(q, sub_regions, dim_to_check);
    for(auto i: axis_lengths_at_dim)
      CHECK(i == Approx(length_pre_split));
        
    dim_to_check = 1;
    axis_lengths_at_dim = check_sub_region_lengths<post_split_num_regions, ndim>(q, sub_regions, dim_to_check);
    for(auto i: axis_lengths_at_dim)
      CHECK(i == Approx(length_post_split));
  }
    
  SECTION("Split a few regions at dim 1, the rest on dim 0"){
    std::fill(sub_dividing_dim.begin(), sub_dividing_dim.end(), 0);
    const std::array<int, 3> regions_with_dim1_split = {1, 105, 309};
    sub_dividing_dim[1] = 1;
    sub_dividing_dim[105] = 1;
    sub_dividing_dim[309] = 1;
    memcpy(classifiers.sub_dividing_dim, sub_dividing_dim.data(),
	       sizeof(int) * pre_split_num_regions);
    splitter.split(q, sub_regions, classifiers);    
        
    for(auto parent_index : regions_with_dim1_split){
      size_t dim_to_check = 1;
      const size_t l_child_reg_id = parent_index;
      const size_t r_child_reg_id = parent_index + pre_split_num_regions;
            
      CHECK(Approx(length_post_split) == sub_region_length_at_dim<ndim>(q, sub_regions, l_child_reg_id, dim_to_check));
      CHECK(Approx(length_post_split) == sub_region_length_at_dim<ndim>(q, sub_regions, r_child_reg_id, dim_to_check));
            
      dim_to_check = 0;
      CHECK(Approx(length_pre_split) == sub_region_length_at_dim<ndim>(q, sub_regions, l_child_reg_id, dim_to_check));
      CHECK(Approx(length_pre_split) == sub_region_length_at_dim<ndim>(q, sub_regions, r_child_reg_id, dim_to_check));
    }
        
    for(int i= 0; i< pre_split_num_regions; ++i){
      if(i != regions_with_dim1_split[0] && i != regions_with_dim1_split[1] && i != regions_with_dim1_split[2]){
	size_t dim_to_check = 0;
	const size_t l_child_reg_id = i;
	const size_t r_child_reg_id = i + pre_split_num_regions;
                
	CHECK(Approx(length_post_split) == sub_region_length_at_dim<ndim>(q, sub_regions, l_child_reg_id, dim_to_check));
	CHECK(Approx(length_post_split) == sub_region_length_at_dim<ndim>(q, sub_regions, r_child_reg_id, dim_to_check));
                 
	dim_to_check = 1;
	CHECK(Approx(length_pre_split) == sub_region_length_at_dim<ndim>(q, sub_regions, l_child_reg_id, dim_to_check));
	CHECK(Approx(length_pre_split) == sub_region_length_at_dim<ndim>(q, sub_regions, r_child_reg_id, dim_to_check));
      }
    }       
  }
    
    
}
