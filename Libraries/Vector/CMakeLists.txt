# Declare the header-only library as an INTERFACE target
add_library(libnumeric_vector STATIC ${CMAKE_CURRENT_SOURCE_DIR}/dummy.cpp)

# Set the include directories
target_include_directories(
  libnumeric_vector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                           ${CMAKE_SOURCE_DIR}/Libraries ${ARMPL_INCLUDE_DIR})

# Link external libraries if needed
target_link_libraries(libnumeric_vector PUBLIC ${ARMPL_BLAS_LIB}
                                               ${ARMPL_LAPACK_LIB})

# Link other libraries here if needed target_link_libraries(libnumeric_vector
# other_library)

enable_testing()
add_executable(test_vector ${CMAKE_CURRENT_SOURCE_DIR}/tests/testVector.cpp)
target_link_libraries(test_vector PRIVATE libnumeric_vector)
add_test(NAME VectorTests COMMAND test_vector)
