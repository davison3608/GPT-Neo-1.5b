#include "embed.h"

namespace py = pybind11;
using std::cerr;
using std::cout;
using namespace Alias;
using namespace Alias::Pb;

/* deser 索引化对象 */
deser::deser(_cstr path, _cstr func_name)
{
    //锁住
    _pygil gil;
    cout<<"尝试导入反索引化脚本 \n";
    try {
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
    cout<<"已导入deser脚本 file: "<<path;
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
    cerr<<"加载deser分词器失败 停止"<<e.what()<<"\n";
    exit(1);
    }    
    cout<<"已加载deser分词器 tokenizer到本地 "<<'\n';

    return ;
}

void deser::getdeser(_vec& index, _cstr& results)
{
    //锁住
    _pygil gil;
    //省略
    return ;
}
