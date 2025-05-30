add_executable(test_rk45
               ${CMAKE_CURRENT_SOURCE_DIR}/tests/harmonic_oscillator.cpp)
target_link_libraries(test_rk45 PRIVATE swnumeric)
add_test(NAME ODEIntegratorTests COMMAND test_rk45)
