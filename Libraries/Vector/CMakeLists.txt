# Link other libraries here if needed target_link_libraries(libnumeric_vector
# other_library)

add_executable(test_vector ${CMAKE_CURRENT_SOURCE_DIR}/tests/testVector.cpp)
target_link_libraries(test_vector PRIVATE swnumeric)
add_test(NAME VectorTests COMMAND test_vector)
