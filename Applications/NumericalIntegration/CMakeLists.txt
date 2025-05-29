add_library(libnumeric_numericalintegration STATIC
            ${CMAKE_CURRENT_SOURCE_DIR}/dummy.cpp)

target_include_directories(
  libnumeric_numericalintegration
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR} ${ARMPL_INCLUDE_DIR})

target_link_libraries(
  libnumeric_numericalintegration PUBLIC ${ARMPL_BLAS_LIB} ${ARMPL_LAPACK_LIB}
                                         libnumeric_vector libnumeric_matrix)

enable_testing()
add_executable(test_gl ${CMAKE_CURRENT_SOURCE_DIR}/tests/testGaussLegendre.cpp)
target_link_libraries(test_gl PRIVATE libnumeric_vector libnumeric_matrix
                                      libnumeric_numericalintegration)
add_test(NAME QuadratureTests COMMAND test_gl)
