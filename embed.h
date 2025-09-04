#ifndef EMBED
#define EMBED
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <pybind11/numpy.h>
#include "strpackets.h"

//! 词汇表长度
extern Alias::_usize vocab_size;
//! 嵌入维度长度
extern Alias::_usize embed_size;

namespace Alias {
    
namespace Pb {
    using _pyfile = pybind11::module;
    using _pyfunc = pybind11::object;
    using _pyobjc = pybind11::object;
    using _pygil = pybind11::gil_scoped_acquire; 

    using _pydict = pybind11::dict;
    using _pytup = pybind11::tuple;
    using _pyarry = pybind11::array_t<precison>;
} // namespace PB

} // namespace Alias

//! 获取当前目录的上级绝对路径
static void getpwd(Alias::_cstr& path);

//! 索引化对象
class INdex
{
protected:
    //! py文件对象   
    Alias::Pb::_pyfile file;
    //! py函数对象
    Alias::Pb::_pyfunc func;

    //! 索引化数据
    Alias::_vec ids;
    Alias::_vec mask;
    //! 本次序列长度
    Alias::_usize sequence;

public:
    INdex(): INdex("index", "string_index")
    {}
    INdex(Alias::_cstr path, Alias::_cstr func_name);
    ~INdex() noexcept = default;

    //! 调用脚本 填充到本地
    void getindex(Alias::_cstr& quest);
    //! 本地调用
    void getindex_cpp(Alias::_cstr& quest);

    //! 返回数据指针的引用
    //! 无线程安全
    Alias::precison *getids_ptr() const noexcept;
    Alias::precison *getmask_ptr() const noexcept;
    //! 返回序列长度
    //! 无线程安全
    Alias::_usize getsequence() const noexcept;

private:
    //! 分词器对象
    Alias::Pb::_pyobjc tokenizer;
    //! numpy包
    Alias::Pb::_pyobjc np;
};

#ifndef DESER_
#define DESER_
//! 反序列化接口对象
class deser
{
protected:
    //! py文件对象   
    Alias::Pb::_pyfile file;
    //! py函数对象
    Alias::Pb::_pyfunc func;

public:
    deser(): deser("deser", "string_deindex")
    {}
    deser(Alias::_cstr path, Alias::_cstr func_name);    
    ~deser() noexcept = default;

    //! 调用脚本解码 gil安全
    void getdeser(Alias::_vec& index, Alias::_cstr& results);
    //! 调用本地
    void getdeser_cpp(Alias::_vec& index, Alias::_cstr& results);

private:
    //! 分词器对象
    Alias::Pb::_pyobjc tokenizer;
    //! numpy包
    Alias::Pb::_pyobjc np;
};

#endif // DESER_ 

#ifndef CSV_
#define CSV_
//! csv写入接口对象
class csvwirte
{
protected:
    //! py文件对象   
    Alias::Pb::_pyfile file;
    //! py函数对象
    Alias::Pb::_pyfunc func;   
    //! csv文件路径
    Alias::_cstr csv_file="../talks_csv.csv"; 

public:
    csvwirte(): csvwirte("csvport", "single_talk")
    {}
    csvwirte(Alias::_cstr path, Alias::_cstr func_name);
    ~csvwirte() noexcept = default;

    using year_month_day_ = std::tuple<int, int, int>;

    //! 写入脚本
    void singlerow_wirte(
        year_month_day_ days,
        Alias::_cstr& quest,
        Alias::_cstr& answer
    );
    //! 本地调用 
    void singlerow_wirte_cpp(
        year_month_day_ days,
        Alias::_cstr& quest,
        Alias::_cstr& answer
    );

private:
    //! numpy包
    Alias::Pb::_pyobjc np;
    //! os包
    Alias::Pb::_pyobjc os;
    //! pandas包
    Alias::Pb::_pyobjc pad;

};

#endif // CSV_ 

#endif // EMBED 