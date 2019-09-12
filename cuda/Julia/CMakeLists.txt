file(GLOB juliaSource *.cpp *.cu)
file(GLOB juliaInclude *.h)

cuda_add_executable(julia ${juliaSource})
target_link_libraries(julia cuda)
