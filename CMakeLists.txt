cmake_minimum_required(VERSION 3.18)
project(talkterminal LANGUAGES CUDA CXX)

# 强制调试模式
set(CMAKE_BUILD_TYPE Debug CACHE STRING "构建类型" FORCE)

# 配置C++17标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译器配置
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# nvcc编译选项
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} \
    -O0 -g -G \
    --expt-relaxed-constexpr \
    --default-stream per-thread \
    --compiler-options -fPIC")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Qt5配置
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5 COMPONENTS 
    Core Gui Qml Quick QuickControls2 
    REQUIRED
)
message(STATUS "找到Qt5: ${Qt5_DIR} (版本: ${Qt5_VERSION})")

# CUDA路径与验证（保持不变）
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include")
if(NOT EXISTS "${CUDA_INCLUDE_DIRS}/cuda_runtime.h" OR NOT EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib/libcudart.so")
    message(FATAL_ERROR "CUDA安装不完整")
endif()

# CUDA架构
set(CUDA_ARCHS 61 86)
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHS})

# 兼容政策
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()

# 第三方依赖配置（保持不变）
set(TVM_HOME $ENV{TVM_HOME})
if(NOT TVM_HOME)
    set(TVM_HOME "$ENV{HOME}/tvm" CACHE PATH "TVM路径" FORCE)
endif()
set(DLPACK_HOME "${TVM_HOME}/3rdparty/dlpack")
set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
if(NOT TENSORRT_ROOT)
    set(TENSORRT_ROOT "/usr/local/TensorRT-8.6.1.6" CACHE PATH "TensorRT路径" FORCE)
endif()

# 依赖查找（保持不变）
find_library(DMLC_LIBRARY NAMES dmlc libdmlc PATHS "/usr/local/lib" "${TVM_HOME}/3rdparty/dmlc-core/build" "${TVM_HOME}/build" REQUIRED)
set(Python_EXECUTABLE "/usr/bin/python3.8" CACHE FILEPATH "Python解释器" FORCE)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED HINTS /usr/local/share/cmake/pybind11 PATHS /usr/local/lib/cmake/pybind11)
set(MySQL_INCLUDE_DIRS "/usr/include/mysql")
set(MySQL_LIBRARIES "/usr/lib/x86_64-linux-gnu/libmysqlclient.so")
foreach(check_path "${MySQL_INCLUDE_DIRS}/mysql.h" "${MySQL_LIBRARIES}" "${DLPACK_HOME}/include/dlpack/dlpack.h")
    if(NOT EXISTS "${check_path}")
        message(FATAL_ERROR "关键依赖缺失: ${check_path}")
    endif()
endforeach()
find_path(TVM_INCLUDE_DIRS NAMES tvm/runtime/c_runtime_api.h PATHS "${TVM_HOME}/include" "/usr/local/include/tvm" REQUIRED)
find_library(TVM_LIBRARY NAMES tvm PATHS "${TVM_HOME}/build" "/usr/local/lib" REQUIRED)
find_library(TVM_RUNTIME_LIBRARY NAMES tvm_runtime PATHS "${TVM_HOME}/build" "/usr/local/lib" REQUIRED)
find_path(TENSORRT_INCLUDE_DIRS NAMES NvInfer.h PATHS "${TENSORRT_ROOT}/include" REQUIRED)
find_library(NVINFER_LIBRARY NAMES nvinfer PATHS "${TENSORRT_ROOT}/lib" REQUIRED)
find_library(NVINFER_PLUGIN_LIBRARY NAMES nvinfer_plugin PATHS "${TENSORRT_ROOT}/lib" REQUIRED)
find_library(NVPARSERS_LIBRARY NAMES nvparsers PATHS "${TENSORRT_ROOT}/lib")
find_library(NVTOOLS_LIBRARY NAMES nvToolsExt PATHS "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib" REQUIRED)

# 包含目录
include_directories(
    include
    ${CUDA_INCLUDE_DIRS}
    "/usr/local/include"
    ${DLPACK_HOME}/include
    ${pybind11_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${MySQL_INCLUDE_DIRS}
    ${TVM_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
    ${Qt5Core_INCLUDE_DIRS}
    ${Qt5Gui_INCLUDE_DIRS}
    ${Qt5Qml_INCLUDE_DIRS}
    ${Qt5Quick_INCLUDE_DIRS}
)

# 链接目录
link_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib
    ${TENSORRT_ROOT}/lib
    ${TVM_HOME}/build
    "/usr/local/lib"
    so/
)

# 源文件列表（明确区分C++和CUDA文件）
set(CXX_FILES
    main.cpp
    threads.cpp
    cachequeue.cpp
    deser.cpp
    hashnodelru.cpp
    process.cpp
    processfunc.cpp
    sqconnect.cpp
    terminal.cpp
    csvwirte.cpp
    embed.cpp
    qtfilewirte.cpp
)
set(CUDA_FILES
    init.cu
    ready.cu
    infer.cu
)
set(HEADER_FILES
    strpackets.h
)

# 强制设置所有C++文件的编译方式（关键修改）
foreach(src ${CXX_FILES})
    set_source_files_properties(${src} PROPERTIES
        LANGUAGE CXX
        COMPILE_FLAGS "-std=c++17"  # 显式指定标准，确保编译
    )
endforeach()

# 设置CUDA文件编译方式
foreach(src ${CUDA_FILES})
    set_source_files_properties(${src} PROPERTIES
        LANGUAGE CUDA
    )
endforeach()

# 合并所有源文件
set(SOURCE_FILES ${CXX_FILES} ${CUDA_FILES} ${HEADER_FILES})

# 添加可执行目标（强制包含所有文件）
add_executable(talkterminal ${SOURCE_FILES})

# 目标属性（关闭分离编译以简化链接）
set_target_properties(talkterminal PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF  # 关键修改：禁用分离编译，避免链接遗漏
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# 链接依赖库
target_link_libraries(talkterminal PRIVATE
    Qt5::Core Qt5::Gui Qt5::Qml Qt5::Quick Qt5::QuickControls2
    pybind11::embed
    ${MySQL_LIBRARIES}
    ${TVM_LIBRARY} ${TVM_RUNTIME_LIBRARY}
    ${DMLC_LIBRARY}
    ${NVINFER_LIBRARY} ${NVINFER_PLUGIN_LIBRARY} ${NVPARSERS_LIBRARY}
    cuda cudart cublas cublasLt cudnn cufft curand nvrtc nvjpeg
    ${NVTOOLS_LIBRARY}
    ${Python_LIBRARIES}
)

# 安装和后处理（保持不变）
install(TARGETS talkterminal RUNTIME DESTINATION bin)
install(DIRECTORY so/ DESTINATION lib/talkterminal)
install(DIRECTORY vocab/ DESTINATION share/talkterminal/vocab)

add_custom_command(TARGET talkterminal POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TVM_LIBRARY}" $<TARGET_FILE_DIR:talkterminal>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${NVINFER_LIBRARY}" $<TARGET_FILE_DIR:talkterminal>
    COMMENT "复制依赖库..."
)

# 配置摘要
message(STATUS "=============================================")
message(STATUS "编译配置摘要:")
message(STATUS "  C++文件: ${CXX_FILES}")  # 验证main.cpp是否在列
message(STATUS "  C++编译器: ${CMAKE_CXX_COMPILER}")
message(STATUS "  CUDA编译器: ${CMAKE_CUDA_COMPILER}")
message(STATUS "=============================================")