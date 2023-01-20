add_executable(Reduction Reduction.cpp)
#set_target_properties(Reduction PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
#target_link_libraries(Reduction util)
target_include_directories(Reduction PRIVATE
  ${CMAKE_SOURCE_DIR}
)
add_test(Reduction Reduction)													

add_executable(finished_estimates finished_estimates.cpp)
#set_target_properties(finished_estimates PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(finished_estimates PUBLIC MKL::MKL_DPCPP)
target_include_directories(finished_estimates PRIVATE
  ${CMAKE_SOURCE_DIR}
)
add_test(finished_estimates finished_estimates)													

add_executable(test_min_max test_min_max.cpp)
target_link_libraries(test_min_max PUBLIC MKL::MKL_DPCPP)
target_include_directories(test_min_max PRIVATE ${CMAKE_SOURCE_DIR})
add_test(test_min_max test_min_max)													

add_executable(test_heuristic_classifer test_heuristic_classifer.cpp)
target_link_libraries(test_heuristic_classifer PUBLIC MKL::MKL_DPCPP)
target_include_directories(test_heuristic_classifer PRIVATE ${CMAKE_SOURCE_DIR})
add_test(test_heuristic_classifer test_heuristic_classifer)													

add_executable(region_filtering region_filtering.cpp)
target_link_libraries(region_filtering PUBLIC MKL::MKL_DPCPP)
target_include_directories(region_filtering PRIVATE ${CMAKE_SOURCE_DIR})
add_test(region_filtering region_filtering)

add_executable(region_splitting region_splitting.cpp)
target_link_libraries(region_splitting PUBLIC MKL::MKL_DPCPP)
target_include_directories(region_splitting PRIVATE ${CMAKE_SOURCE_DIR})
add_test(region_splitting region_splitting)							

add_executable(exclusive_parallel_scan exclusive_parallel_scan.cpp)
target_link_libraries(exclusive_parallel_scan PUBLIC MKL::MKL_DPCPP)
target_include_directories(exclusive_parallel_scan PRIVATE ${CMAKE_SOURCE_DIR})
add_test(exclusive_parallel_scan exclusive_parallel_scan)							
