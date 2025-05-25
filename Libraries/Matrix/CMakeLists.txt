add_library(libnumeric_matrix STATIC ${CMAKE_CURRENT_SOURCE_DIR}/dummy.cpp)

# Set the include directories
target_include_directories(
  libnumeric_matrix PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}
                           ${ARMPL_INCLUDE_DIR})

# Link external libraries if needed
target_link_libraries(
  libnumeric_matrix PUBLIC ${ARMPL_BLAS_LIB} ${ARMPL_LAPACK_LIB}
                           libnumeric_vector)

# Link other libraries here if needed target_link_libraries(libnumeric_vector
# other_library)
#
enable_testing()
add_executable(test_matrix ${CMAKE_CURRENT_SOURCE_DIR}/tests/testMatrix.cpp)
target_link_libraries(test_matrix PRIVATE libnumeric_matrix)
add_test(NAME VectorTests COMMAND test_matrix)
