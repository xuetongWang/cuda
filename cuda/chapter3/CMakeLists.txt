file(GLOB chapterS *.cpp *.cu)
file(GLOB chapterI *.h)

cuda_add_executable(chapter3 ${chapterS})
target_link_libraries(chapter3 cuda)
