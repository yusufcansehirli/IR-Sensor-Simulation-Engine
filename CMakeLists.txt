cmake_minimum_required(VERSION 3.20)
project(IRSensorSim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─── Build type default ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# ─── CUDA settings ─────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)   # Turing / Ampere / Ada

# FetchContent tüm bloklar için
include(FetchContent)

# ─── OpenGL ────────────────────────────────────────────────────────────────────
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

# GLAD — FetchContent ile v0.1.36 (single-file C generator)
# glad_dl FetchContent'i kendi 'glad' cmake targetını ekler
FetchContent_Declare(glad_src
  GIT_REPOSITORY https://github.com/Dav1dde/glad.git
  GIT_TAG        v0.1.36
)
FetchContent_MakeAvailable(glad_src)
# FetchContent sonrası glad hedefi mevcut — dolaylı olarak glad_src_SOURCE_DIR GLAD dosyalarını üretir
# Python generator kullanarak dosyaları oluştur
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(GLAD_DIR "${CMAKE_BINARY_DIR}/glad_generated")
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m glad
    --generator c --spec gl --out-path "${GLAD_DIR}"
    --api "gl:core=4.5" --extensions ""
  WORKING_DIRECTORY "${glad_src_SOURCE_DIR}"
  RESULT_VARIABLE GLAD_GEN_RESULT
  OUTPUT_QUIET
  ERROR_QUIET
)
if(NOT GLAD_GEN_RESULT EQUAL 0)
  # Fallback: doğrudan glad_src kaynak dizinini kullan
  message(STATUS "GLAD Python generator başarısız, doğrudan kaynak kullanılıyor")
  set(GLAD_DIR "${glad_src_SOURCE_DIR}")
endif()
# GLAD kaynak dosyası eğer varsa (generator çıktısı)
if(EXISTS "${GLAD_DIR}/src/glad.c")
  add_library(ir_glad STATIC "${GLAD_DIR}/src/glad.c")
  target_include_directories(ir_glad PUBLIC "${GLAD_DIR}/include")
  set_target_properties(ir_glad PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
  # FetchContent glad target'ı kullan
  add_library(ir_glad ALIAS glad)
  set_target_properties(glad PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set(GLAD_DIR "${glad_src_SOURCE_DIR}")
endif()


# ─── yaml-cpp ──────────────────────────────────────────────────────────────────
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
  FetchContent_Declare(yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        0.8.0
  )
  set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(yaml-cpp)
endif()

# ─── HDF5 ──────────────────────────────────────────────────────────────────────
find_package(HDF5 COMPONENTS C CXX QUIET)

# ─── GLM ───────────────────────────────────────────────────────────────────────
find_package(glm QUIET)
if(NOT glm_FOUND)
  include(FetchContent)
  FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        0.9.9.8
  )
  FetchContent_MakeAvailable(glm)
endif()

# ─── ImGui (FetchContent) ─────────────────────────────────────────────────────
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        v1.90.4
)
FetchContent_MakeAvailable(imgui)
set(IMGUI_DIR "${imgui_SOURCE_DIR}")

set(IMGUI_SOURCES
  "${IMGUI_DIR}/imgui.cpp"
  "${IMGUI_DIR}/imgui_draw.cpp"
  "${IMGUI_DIR}/imgui_tables.cpp"
  "${IMGUI_DIR}/imgui_widgets.cpp"
  "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
  "${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp"
)

# ─── Shared sources ────────────────────────────────────────────────────────────

set(ENGINE_SOURCES
  src/ir_engine.cpp
  src/material_db.cpp
  cuda/planck_kernel.cu
  cuda/thermal_solver.cu
  cuda/radiance_kernel.cu
  cuda/atm_lut.cu
  cuda/mtf_kernel.cu
  cuda/sensor_model.cu
)

# ─── Main executable ───────────────────────────────────────────────────────────
add_executable(ir_sim
  src/main.cpp
  ${ENGINE_SOURCES}
  ${IMGUI_SOURCES}
)

target_include_directories(ir_sim PRIVATE
  include
  "${IMGUI_DIR}"
  "${IMGUI_DIR}/backends"
  "${GLAD_DIR}/include"
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(ir_sim PRIVATE
  ir_glad
  OpenGL::GL
  glfw
  yaml-cpp
  glm::glm
  CUDA::cudart
  CUDA::curand
  ${HDF5_LIBRARIES}
)

if(HDF5_FOUND)
  target_include_directories(ir_sim PRIVATE ${HDF5_INCLUDE_DIRS})
  target_compile_definitions(ir_sim PRIVATE HAVE_HDF5)
endif()

target_compile_definitions(ir_sim PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLAD)

# CUDA separable compilation for device function linking
set_target_properties(ir_sim PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ─── Shared plugin library (for Unity / UE5) ───────────────────────────────────
add_library(ir_sensor_plugin SHARED
  src/ir_plugin_api.cpp
  ${ENGINE_SOURCES}
)

target_include_directories(ir_sensor_plugin PRIVATE
  include
  "${GLAD_DIR}/include"
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(ir_sensor_plugin PRIVATE
  ir_glad
  OpenGL::GL
  yaml-cpp
  glm::glm
  CUDA::cudart
  CUDA::curand
  ${HDF5_LIBRARIES}
)

if(HDF5_FOUND)
  target_include_directories(ir_sensor_plugin PRIVATE ${HDF5_INCLUDE_DIRS})
  target_compile_definitions(ir_sensor_plugin PRIVATE HAVE_HDF5)
endif()

set_target_properties(ir_sensor_plugin PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ─── Tests ─────────────────────────────────────────────────────────────────────
add_executable(test_physics
  tests/test_physics.cpp
  src/material_db.cpp
  cuda/planck_kernel.cu
)

target_include_directories(test_physics PRIVATE
  include
  "${GLAD_DIR}/include"
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(test_physics PRIVATE
  yaml-cpp
  CUDA::cudart
)

set_target_properties(test_physics PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# ─── Install data files ────────────────────────────────────────────────────────
file(COPY "${CMAKE_SOURCE_DIR}/data"    DESTINATION "${CMAKE_BINARY_DIR}")
file(COPY "${CMAKE_SOURCE_DIR}/shaders" DESTINATION "${CMAKE_BINARY_DIR}")
