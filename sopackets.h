#ifndef SOLOAD
#define SOLOAD
#include "iostream"
#include "assert.h"
#include "string"

#define TRTFILE
#define SOFILE

#include <cuda_runtime.h>
#include <cuda.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>

//运行时相关
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/ndarray.h>
//设备上下文相关
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/c_runtime_api.h>
//张量表达式与计算图构建相关
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>

//! 通用检查
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d, code=%d(%s)\n", __FILE__, __LINE__, \
               err, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

class logger: public nvinfer1::ILogger
{
public:
    logger() = default;
    ~logger() = default;

    //重写虚函数
    void log(Severity severity, const char* msg) noexcept override 
    {
    if(severity !=Severity::kINFO 
    && severity != Severity::kWARNING 
    && severity != Severity::kERROR 
    && severity != Severity::kINTERNAL_ERROR
    && severity != Severity::kVERBOSE)
        return;
    switch (severity)
    {
    case Severity::kINFO:
    printf("kINFO info is: %s\n",msg);
        break;
    case Severity::kWARNING:
    printf("kWARNING warning is: %s\n",msg);
        break;
    case Severity::kERROR:
    printf("kERROR error is: %s\n",msg);
        break;
    case Severity::kINTERNAL_ERROR:
    printf("kINTERNAL_ERROR internal error is: %s\n",msg);
        break;
    case Severity::kVERBOSE:
    printf("kVERBOSE is: %s\n",msg);
    default:
        break;
    }
    }
};

#ifdef TRTFILE
//! 加载trt文件
//! 自动释放
class RTload
{
protected:
    //！引擎对象
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IRuntime *runtime;
    logger log;

public:
    //! 加载运行时
    //! 加载引擎文件
    RTload(): RTload(
        "../engine/1.5b-tensorrt-half-256sequence.trt"
    )
    {}
    RTload(std::string path);
    //! 销毁句柄
    ~RTload();
};

#endif //TRTFILE

#include "strpackets.h"

namespace Alias {

namespace VM {
    using _vmdevice = DLDevice;
    using _vmmod = tvm::runtime::Module;
    using _vmexec = tvm::runtime::Module;
    using _vmparam = std::vector<char>;
    using _tensor = DLTensor;
} // namespace VM

} // namespace Alias

#ifdef SOFILE
//! 加载so文件
//! 自动释放
class Soload
{
    //! so文件 json文件 权重数据
    Alias::VM::_vmmod so_file;
    Alias::VM::_vmparam json_file;
    Alias::VM::_vmparam weights;
    //! 设备信息
    Alias::VM::_vmdevice dev;

public:
    using _cstr = std::string;
    //! 加载权重数据 生命与Soload绑定
    Soload(): Soload(
        "../so/graph.json", 
        "../so/1.5b-tvm-llvm-float32-100sequence.so", 
        "../so/weights.params"
    )
    {} 
    //! 计算图路径 so路径 权重路径 
    Soload(_cstr json_ph, _cstr so_ph, _cstr params_ph);
    //! 无需析构
    ~Soload();
};

#endif //SOFILE

#endif //SOLOAD