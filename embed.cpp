#include "embed.h"

namespace py = pybind11;
using std::cerr;
using std::cout;
using namespace Alias;
using namespace Alias::Pb;

Alias::_usize vocab_size = 50257;
Alias::_usize embed_size = 400;

void getpwd(Alias::_cstr& path)
{
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
    // build目录上一级目录
    _cstr current_dir(cwd);
    _size last_slash=current_dir.find_last_of("/\\");
    // 获取上级目录
    if (last_slash != std::string::npos) {
    path = current_dir.substr(0, last_slash);
    } 
    // 处理异常路径情况
    else 
    path = current_dir;
    
    } 
    else {
    // 获取路径失败时的错误处理
    cerr << "获取当前目录失败: " << strerror(errno) << "\n";
    exit(1);
    }
    return ;
}

/* INdex 索引化对象 */
INdex::INdex(_cstr path, _cstr func_name)
{
    //锁住
    _pygil gil;
    cout<<"尝试导入分词脚本 \n";
    try {
    //获取脚本绝对路径
    _cstr dirpath;
    getpwd(dirpath);
    //path=dirpath + "/" +path;

    //加载文件
    this->file=py::module::import(path.data());
    //加载函数
    this->func=this->file.attr(func_name.data());   
    }
    catch(const std::exception& e) {
    cerr<<"加载py文件与函数失败 停止"<<e.what()<<"\n";
    exit(1);
    }
    catch (const py::error_already_set& e) {
    cerr<<"加载py文件与函数失败 停止"<<e.what()<<"\n";
    exit(1);
    }
    cout<<"已导入分词脚本 file: "<<path;
    cout<<" func: "<<func_name<<'\n';

    //加载软件包
    try {
    this->np=py::module::import("numpy");
    }
    catch (const py::error_already_set& e) {
    cerr<<"加载numpy失败 停止"<<e.what()<<"\n";
    exit(1);
    }
    try {
    //transformers软件包
    py::module transformers=pybind11::module::import("transformers");
    //加载分词器
    this->tokenizer=transformers.attr("AutoTokenizer").attr("from_pretrained")("../vocab");
    //设置填充
    if (this->tokenizer.attr("pad_token_id").is_none())
        this->tokenizer.attr("pad_token_id")=this->tokenizer.attr("eos_token_id");
    }
    catch (const std::exception& e) {
    cerr<<"加载INdex分词器失败 停止"<<e.what()<<"\n";
    exit(1);
    }    
    cout<<"已加载INdex分词器 tokenizer到本地 "<<'\n';

    return ;
}

void INdex::getindex(_cstr& quest)
{
    //提前声明元祖
    _pytup results;
    {
    //锁住
    _pygil gil;
    //调用
    results=this->func(quest);
    }

    //已释放gil 填充到本地
    _pyarry one=results[0].cast<_pyarry>();
    _pyarry two=results[1].cast<_pyarry>();
    
    auto shapes=one.shape();
    auto batch=shapes[0];
    auto sequence=shapes[1];
    if (batch != 1 || sequence != two.shape()[1])
    { cerr<<"分词有误 本次不填充\n"; return ; }

    this->sequence=sequence;

    this->ids.resize(sequence);
    this->mask.resize(sequence);

    //二维访问
    auto id_data=one.unchecked<2>();
    auto ma_data=two.unchecked<2>();
    for (int i=0; i<batch; i++) {
    for (int j=0; j<sequence; j++) {
        this->ids[j]=id_data(i, j);
        this->mask[j]=ma_data(i, j);
    }
    }
}

precison *INdex::getids_ptr() const noexcept
{ 
    return const_cast<precison*>(this->ids.data()); 
}

precison *INdex::getmask_ptr() const noexcept
{ 
    return const_cast<precison*>(this->mask.data()); 
}

_usize INdex::getsequence() const noexcept
{ 
    return this->sequence; 
}

void INdex::getindex_cpp(_cstr& quest)
{
    //锁住
    _pygil gil;

    //分词方法
    py::dict results=this->tokenizer.attr("__call__")(
        quest,
        pybind11::arg("return_tensors")="np"
    );
    
    //获取索引与掩码
    const _cstr res_0="input_ids";
    const _cstr res_1="attention_mask";

    if (results.contains(res_0) && results.contains(res_1)) {
    //一维索引 int64精度
    py::array_t<precison> ids_=results[res_0.data()].
        attr("squeeze")(0).
        attr("astype")("int64");
    py::array_t<precison> mask_=results[res_1.data()].
        attr("squeeze")(0).
        attr("astype")("int64");

    //获取分词长度 
    this->sequence=ids_.size();
    if (this->sequence != mask_.size())
    { cerr<<"分词错误 掩码与索引长度不一致 已返回 !\n"; return ; }

    //重新置0
    this->ids.resize(this->sequence);
    this->mask.resize(this->sequence);
    //深拷贝副本
    py::buffer_info ids_bf=ids_.request();
    py::buffer_info mask_bf=mask_.request();
    auto ids_p=static_cast<precison*>(ids_bf.ptr);
    auto mask_p=static_cast<precison*>(mask_bf.ptr);

    std::copy(
        ids_p,
        ids_p + this->sequence,
        this->ids.begin()
    );
    std::copy(
        mask_p,
        mask_p + this->sequence,
        this->mask.begin()
    );
        return ;
    }
    else {
        cerr<<"分词错误 无字典键 已返回 !\n";
        return ;
    }
}