file(GLOB vectorSource *.cpp *.cu)
file(GLOB vectorInclude *.h)

cuda_add_executable(vector ${vectorSource})
target_link_libraries(vector cuda)
