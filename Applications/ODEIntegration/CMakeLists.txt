add_library(libnumeric_odeintegration STATIC
            ${CMAKE_CURRENT_SOURCE_DIR}/RungeKutta45.cpp)

# Set the include directories
target_include_directories(
  libnumeric_odeintegration PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                   ${CMAKE_SOURCE_DIR} ${ARMPL_INCLUDE_DIR})

# Link external libraries if needed
target_link_libraries(
  libnumeric_odeintegration PUBLIC ${ARMPL_BLAS_LIB} ${ARMPL_LAPACK_LIB}
                                   libnumeric_vector libnumeric_matrix)

enable_testing()
add_executable(test_rk45
               ${CMAKE_CURRENT_SOURCE_DIR}/tests/harmonic_oscillator.cpp)
target_link_libraries(test_rk45 PRIVATE libnumeric_vector libnumeric_matrix
                                        libnumeric_odeintegration)
add_test(NAME ODEIntegratorTests COMMAND test_rk45)
