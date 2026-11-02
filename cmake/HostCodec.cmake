option(TINYC_HOST_CODEC "Build the host quantization library used by weight export" OFF)
if(TINYC_HOST_CODEC)
  add_subdirectory(backend/host_codec)
endif()
