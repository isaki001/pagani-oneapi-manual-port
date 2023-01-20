#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#define CATCH_CONFIG_MAIN
#include "externals/catch2/catch.hpp"
#include <CL/sycl.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
//#include "oneapi/mkl.hpp"
#include "oneAPI/quad/GPUquad/Sample.h"
using namespace sycl;

void init_array(double* arr, size_t size){
    for(int i = 0; i < size; ++i){
        arr[i] = 1. + static_cast<double>(i);
    }
}

TEST_CASE("DPL's reduction for host-side operations small array")
{
    constexpr size_t size = 5;
    sycl::queue q;
    std::array<double, size> arr;
    std::fill(arr.begin(), arr.end(), -1);
    
    double* input = sycl::malloc_shared<double>(size, q);
    input = quad::copy_to_shared<double>(q, arr.data(), size);
    double res =  quad::reduction<double>(q, input, size);
    CHECK(res == -5.);
}

TEST_CASE("DPL's reduction for host-side operations large array")
{
    constexpr size_t size = 500;
    sycl::queue q;
    std::array<double, size> arr;
    std::fill(arr.begin(), arr.end(), -1);
    
    double* input = sycl::malloc_shared<double>(size, q);
    input = quad::copy_to_shared<double>(q, arr.data(), size);
    double res =  quad::reduction<double>(q, input, size);
    CHECK(res == -500.);
}


TEST_CASE("Manual Single Block, Single Warp Reduction")
{

sycl::queue q;

  ShowDevice(q);  
    
  double* results = sycl::malloc_shared<double>(32, q);    
  double* input = sycl::malloc_shared<double>(32, q);  
    
  std::cout << "Local Memory Size: "
            << q.get_device().get_info<sycl::info::device::local_mem_size>()
            << std::endl;
    
  //initialize array to reduce  
  for(int i = 0; i < 32; ++i){
      input[i] = 1. + static_cast<double>(i);
  }
    
  static constexpr size_t reduction_array_size = 32;  
  static constexpr size_t work_group_size = 32;
  
  //SECTION("Manual Reduce input array")
  //{
     q.submit([&](auto &cgh) {
     sycl::stream str(8192, 1024, cgh);
     auto sg_sizes = q.get_device().get_info<info::device::sub_group_sizes>();
     for(auto sg_size : sg_sizes)
         std::cout<< "size:"<< sg_size << std::endl;
     cgh.parallel_for(sycl::nd_range<1>(reduction_array_size, work_group_size), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(work_group_size)]] {
       auto sg = item.get_sub_group();
         
       size_t g_id = item.get_global_id()[0];
       size_t sg_id = sg.get_group_id()[0];  
       size_t l_id = sg.get_local_id()[0];
       
       results[g_id] = reduce_over_group(sg, input[l_id], plus<>()); 
         
       str << "edo global thread id "<< g_id 
           << " from group " << sg_id 
           << " local thread id " << l_id 
           << " item.get_local_range:"<< item.get_local_range()[0]
           << " input value:"<< input[g_id] 
           //<< " item.get_group_range:"<<item.get_group_range()[0]
           //<< " sg.get_group_range:"<<sg.get_group_range()[0]
           << sycl::endl;         
         
     });
   }).wait();  
      
      
     for(size_t i =0; i < 32; ++i)
        std::cout<<"edo results "<< i  << " :"<< results[i] << std::endl;
      
    for(size_t i =0; i < 32; ++i){
        CHECK(results[i] == Approx(528.));
    }
  //}
    
    
  /*SECTION("Manual Reduce local variable")
  {
     q.submit([&](auto &cgh) {
     sycl::stream str(8192, 1024, cgh);
     cgh.parallel_for(sycl::nd_range<1>(reduction_array_size, work_group_size), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(32)]] {
       auto sg = item.get_sub_group();
         
       size_t g_id = item.get_global_id();
       size_t sub_group_id = sg.get_group_id()[0];  
       size_t l_id = sg.get_local_id()[0];
       
       double val = 1.5;
       results[g_id] = reduce_over_group(sg, val, plus<>());   
     });
   }).wait();  
    
    for(size_t i =0; i < 32; ++i){
        CHECK(results[i] == Approx(48.));
    }
  }  */
}

TEST_CASE("Manual Two-warp block reduction, result only for thread 0")
{
  sycl::queue q;  
  ShowDevice(q);  
    
  double* results = malloc_shared<double>(64, q);    
  double* input = malloc_shared<double>(64, q);  
  std::cout << "Local Memory Size: "
            << q.get_device().get_info<sycl::info::device::local_mem_size>()
            << std::endl;
    
  //initialize array to reduce  
  for(int i = 0; i < 64; ++i){
      input[i] = 1. + static_cast<double>(i);
  }
    
  size_t reduction_array_size = 64;  
  size_t work_group_size = 64;
  size_t num_sub_groups = work_group_size/32;
  
  SECTION("Reduce input array")
  {
     q.submit([&](auto &cgh) {
         
        sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> sdata(num_sub_groups, cgh);
    
        sycl::stream str(8192, 1024, cgh);
        cgh.parallel_for(sycl::nd_range<1>(reduction_array_size, work_group_size), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(32)]] {
            auto sg = item.get_sub_group();
         
           size_t g_id = item.get_global_id();
           size_t sub_group_id = sg.get_group_id()[0];  
           size_t l_id = sg.get_local_id()[0];
       
           //does oneAPI offer a way to reduce across work-group without supplying a shared mem array and doing manual work?
           //is reduce_group_better than oneAPI's equivalent of shlf_donw_sync?
           double val = reduce_over_group(sg, input[g_id], plus<>());
            
           if(l_id == 0){
               sdata[sub_group_id] = val;
           }
         
           item.barrier(sycl::access::fence_space::local_space);
           val = sub_group_id == 0 ? sdata[l_id] : 0; //only warp 0 writes to val
           
           if(sub_group_id == 0){
               results[g_id] = reduce_over_group(sg, val, plus<>());   
           }  
         
        });
   }).wait();  
    
      //results valid on first warp only
    for(size_t i =0; i < 32; ++i){
        CHECK(results[i] == Approx(2080.));
    }
  }
}


TEST_CASE("Test block-reduction method")
{
    sycl::queue q;  
    ShowDevice(q);  
    printf("in test case\n");
    
    double* input = sycl::malloc_shared<double>(128, q);  
    double* results = malloc_shared<double>(2, q);  
    
    for(int i = 0; i < 128; ++i){
      input[i] = 1. + static_cast<double>(i);
    }
    
    SECTION("One block two warps")
    {
      printf("outside of q submit\n");
        q.submit([&](auto &cgh) {
    	    printf("inside first q submit\n");
            sycl::stream str(8192, 1024, cgh);
        
            size_t reduction_array_size = 64;  
            size_t work_group_size = 64;
            const size_t warp_size = 32;
            size_t num_sub_groups = work_group_size/warp_size;
        
            sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> sdata(sycl::range(num_sub_groups), cgh);
    	    printf("A before parallel for\n");
            cgh.parallel_for(sycl::nd_range<1>(reduction_array_size, work_group_size), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(warp_size)]] {
    		str<< "A inside loop"<<sycl::endl;
                auto sg = item.get_sub_group();
                size_t g_id = item.get_global_id();
                
                double val = quad::block_reduce<double, 64>(item, input[g_id], sdata, str);
                
                if(g_id == 0)
                    results[g_id] = val;
            });
       }).wait();  
    
        CHECK(results[0] == Approx(2080.));
    }
    
    SECTION("Two blocks of two warps each")
    {
      for(int i = 0; i < 128; ++i){
      input[i] = 1. + static_cast<double>(i);
    }
    
      printf("outside of q submit\n");
        q.submit([=](auto &cgh) {
            sycl::stream str(8192, 8192, cgh);
	    //sycl::stream str(64, 64, cgh);
            size_t reduction_array_size = 128;  
            size_t work_group_size = 64;
            const size_t warp_size = 32;
            size_t num_sub_groups = work_group_size/warp_size;
	    
            sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> sdata(num_sub_groups, cgh);
    	    //printf("before parallel for\n");
            cgh.parallel_for(sycl::nd_range<1>(reduction_array_size, work_group_size), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(warp_size)]] {
    		
                auto sg = item.get_sub_group();
                auto wg = item.get_group();
                size_t global_tid = item.get_global_id();
                size_t work_group_tid = item.get_local_id();
                size_t sub_group_id = sg.get_group_id()[0];  
                size_t warp_tid = sg.get_local_id()[0];
                size_t work_group_id = item.get_group_linear_id();
                double val = quad::block_reduce<double, 64>(item, input[global_tid], sdata, str);
                item.barrier(sycl::access::fence_space::local_space);
                str <<"wg_id " << work_group_id
		    << " wg tid " << work_group_tid
		    << " g_tid "<< global_tid
		  //<< " sub_group_id " << sub_group_id
		  //  << " warp_tid " << warp_tid
		    << " val:" << val
		    << " input " << input[global_tid]
		  
		<< sycl::endl;    
                
                if(work_group_tid == 0)
                    results[work_group_id] = val;
            });
       }).wait();  
        
        CHECK(results[0] == Approx(2080.));
        CHECK(results[1] == Approx(6176.));
    }
}


TEST_CASE("Test block-reduction method with thread-local array")
{
    sycl::queue q;  
    ShowDevice(q);  
    
    
    double* input = malloc_shared<double>(128, q);  
    double* results = malloc_shared<double>(64, q);  
    
    for(int i = 0; i < 128; ++i){
      input[i] = 1. + static_cast<double>(i);
    }
    
    SECTION("One block two warps")
    {
        q.submit([&](auto &cgh) {
            sycl::stream str(8192, 1024, cgh);
        
            size_t reduction_array_size = 64;  
            size_t work_group_size = 64;
            const size_t warp_size = 32;
            size_t num_sub_groups = work_group_size/warp_size;
        
            sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> sdata(sycl::range(num_sub_groups), cgh);
        
            cgh.parallel_for(sycl::nd_range<1>(reduction_array_size, work_group_size), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(warp_size)]] {
                const size_t work_group_tid = item.get_local_id();
                const size_t work_group_id = item.get_group_linear_id();
                auto sg = item.get_sub_group();
                size_t g_id = item.get_global_id();
                
                double sum[5] = {0.};
                sum[0] = static_cast<double>(work_group_tid);
                item.barrier(sycl::access::fence_space::local_space);
                
                sum[0] = quad::block_reduce<double, 64>(item, sum[0], sdata, str);
                
                item.barrier(sycl::access::fence_space::local_space);
                
                if(work_group_tid == 0)
                    results[work_group_tid] = sum[0];
            });
       }).wait();  
    
        CHECK(results[0] == Approx(2016.));
    }
}

//add reduction of device array on host

TEST_CASE("Repeated reductions")
{
    sycl::queue q;  
    ShowDevice(q);  
    
    
    double* input = malloc_shared<double>(128, q);  
    double* results = malloc_shared<double>(64, q);  
    
    for(int i = 0; i < 128; ++i){
      input[i] = 1. + static_cast<double>(i);
    }
    
    SECTION("One block two warps")
    {
        q.submit([&](auto &cgh) {
            sycl::stream str(8192, 1024, cgh);
        
            size_t reduction_array_size = 64;  
            size_t work_group_size = 64;
            const size_t warp_size = 32;
            size_t num_sub_groups = work_group_size/warp_size;
        
            sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> sdata(sycl::range(num_sub_groups), cgh);
        
            cgh.parallel_for(sycl::nd_range<1>(reduction_array_size, work_group_size), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(warp_size)]] {
                const size_t work_group_tid = item.get_local_id();
                const size_t work_group_id = item.get_group_linear_id();
                auto sg = item.get_sub_group();
                size_t g_id = item.get_global_id();
                
                double sum[5] = {0.};
                for(int i = 0; i < 5; ++i)
                    sum[i] = static_cast<double>(work_group_tid);
                
                for(int i=0; i < 5; ++i){
                    item.barrier(sycl::access::fence_space::local_space);

                    sum[i] = quad::block_reduce<double, 64>(item, sum[i], sdata, str);

                    item.barrier(sycl::access::fence_space::local_space);
                }
                
                if(work_group_tid == 0){
                    for(int i=0; i < 5; ++i)
                        results[i] = sum[i];
                }
            });
       }).wait();  
        
        for(int i=0; i < 5; ++i)
            CHECK(results[i] == Approx(2016.));
    }
}
