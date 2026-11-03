option(TINYC_SERVER "Build the Linux epoll serving runtime" OFF)
if(TINYC_SERVER)
  add_subdirectory(server)
endif()
