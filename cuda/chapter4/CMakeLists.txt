file(GLOB chapterS *.cpp *.cu)
file(GLOB chapterI *.h)

cuda_add_executable(chapter4 ${chapterS})
target_link_libraries(chapter4 cuda)
