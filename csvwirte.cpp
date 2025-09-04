#include "embed.h"

namespace py = pybind11;
using std::cerr;
using std::cout;
using namespace Alias;
using namespace Alias::Pb;

/* csvwirte csv写入对象 */
csvwirte::csvwirte(_cstr path, _cstr func_name)
{
    //锁住
    _pygil gil;
    cout<<"尝试导入csv脚本 \n";
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
    cout<<"已导入csv脚本 file: "<<path;
    cout<<" func: "<<func_name<<'\n';

    //加载软件包
    try {
    this->np=py::module::import("numpy");
    this->os=py::module::import("os");
    this->pad=py::module::import("pandas");
    }
    catch (const py::error_already_set& e) {
    cerr<<"加载py文件与函数失败 停止"<<e.what()<<"\n";
    exit(1);
    }
    cout<<"已加载csvwirte pandas到本地 "<<'\n';

    return ;
}

void csvwirte::singlerow_wirte(year_month_day_ days,_cstr& quest, _cstr& answer)
{
    //锁住
    _pygil gil;
    //省略
    return ;
}

void csvwirte::singlerow_wirte_cpp(year_month_day_ days, _cstr& quest, _cstr& answer)
{
    //锁住
    _pygil gil;

    using std::get;
    int year=get<0>(days);
    int month=get<1>(days);
    int day=get<2>(days);

    //新的一行 字典
    py::dict new_row;
    new_row["year"]=year;
    new_row["month"]=month;
    new_row["day"]=day;
    new_row["question"]=quest;
    new_row["answer"]=answer;

    try {
    //包装成csv一行
    auto csv_row=this->pad.attr("DataFrame")(py::list{new_row});

    //是否存在 
    bool file_exists=this->os.attr("path").
        attr("exists")(this->csv_file.c_str()).
        cast<bool>();
    
    csv_row.attr("to_csv")(
        this->csv_file.c_str(),
        py::arg("mode")="a", //追加模式
        py::arg("header")=! file_exists, //无则创建csv文件
        py::arg("index")=false //不写入行索引
    );
    }
    catch (const std::exception& e) {
    cerr<<"写入csv文件失败 已返回"<<e.what()<<"\n";
    return ;
    }
}