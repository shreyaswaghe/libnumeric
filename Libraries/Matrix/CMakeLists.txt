# Link other libraries here if needed target_link_libraries(libnumeric_vector
# other_library)
#
add_executable(test_matrix ${CMAKE_CURRENT_SOURCE_DIR}/tests/testMatrix.cpp)
target_link_libraries(test_matrix PRIVATE swnumeric)
add_test(NAME VectorTests COMMAND test_matrix)
