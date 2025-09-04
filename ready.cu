#include "strpackets.h"
#include "sopackets.h"

using std::cerr;
using std::ifstream;
using std::ofstream;
using namespace Alias;

/* TRTFILE 加载trt文件 */
RTload::RTload(std::string path)
{
    ifstream file(path, std::ios::binary);
    assert(file.is_open());

    file.seekg(0, std::ios::end);
    auto file_size=file.tellg();
    file.seekg(0, std::ios::beg);

    using namespace nvinfer1;
    CUDA_CHECK(cudaSetDevice(0));
    this->runtime=createInferRuntime(log);

    std::vector<char> datas;
    datas.resize(file_size);
    file.read(datas.data(), file_size);

    this->runtime->setMaxThreads(2);
    engine=runtime->deserializeCudaEngine(datas.data(), file_size);
    assert(this->engine);
    
    file.close();
    cerr<<"已反序列化trt引擎 "<<path<<'\n';
}

RTload::~RTload()
{
    this->engine->destroy();
    this->runtime->destroy();
}

/* SOFILE 加载so文件 */
Soload::Soload(_cstr json_ph, _cstr so_ph, _cstr params_ph)
{
    //加载so
    ifstream so(so_ph);
    assert(so.is_open());
    so.close();

    using namespace tvm;
    using std::ios;
    this->so_file=runtime::Module::LoadFromFile(so_ph);

    //加载json weights
    ifstream js(json_ph, ios::binary);
    ifstream wg(params_ph, ios::binary);
    assert(js.is_open() && wg.is_open());

    js.seekg(0, ios::end);
    wg.seekg(0, ios::end);
    auto js_size=js.tellg();
    auto wg_size=wg.tellg();
    js.seekg(0, ios::beg);
    wg.seekg(0, ios::beg);

    json_file.resize(js_size);
    weights.resize(wg_size);
    js.read(this->json_file.data(), js_size);
    wg.read(this->weights.data(), wg_size);

    js.close();
    wg.close();

    //设置设备信息
    this->dev.device_type=DLDeviceType::kDLCPU;
    this->dev.device_id=int32_t(0);

    cerr<<"已反序列化tvm文件 "<<so_ph<<'\n';
}

Soload::~Soload()
{}
