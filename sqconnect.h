#ifndef SQLCON
#define SQLCON
#include <mysql.h>
#include <mysql_com.h>
#include <mysql/mysql.h>
#include <mysql/mysql_com.h>

#include "strpackets.h"

namespace Alias {

namespace SQ {
    using _info = std::string;
    using _port = int32_t;

    using _sql = MYSQL;
    using _sql_cmd = const char*;

    using _sql_res = MYSQL_RES;
    using _sql_rows = MYSQL_ROWS;
    using _sql_r = MYSQL_ROW;
} // namespace TypeAlias

} // namespace TypeAlias

extern const Alias::SQ::_info host_;
extern const Alias::SQ::_info user_;
extern const Alias::SQ::_info pass_;
extern Alias::SQ::_info sql_name;
extern Alias::SQ::_port port_;

//! sq全局操作锁
extern Alias::_pmtx pmt_sq;
extern Alias::_pmtx_attr pmt_sq_attr;
//! sq全局init次数
extern Alias::_usize sq_ref;

//! 信息
extern const Alias::SQ::_info host_;
extern const Alias::SQ::_info user_;
extern const Alias::SQ::_info pass_;

extern Alias::SQ::_sql *conn_;
extern Alias::SQ::_port port_;

//! 结构体形式
struct Sqdata
{
    //! 问题
    Alias::SQ::_info quest;
    //! 回答
    Alias::SQ::_info answer;

    //! 插入日期
    Alias::SQ::_info year;
    Alias::SQ::_info month;
    Alias::SQ::_info day;
};

//! 连接sql
//! 静态对象成员
class Sqlite
{ 
public:
    Sqlite() noexcept = default;
    ~Sqlite() noexcept = default;

    //! 手动处理初始化
    static void init_connect(); 

    //! 手动断开
    static void destroy();
};
//! sql增查部分
class SqliteCR: public Sqlite
{
public:
    SqliteCR() noexcept = default;
    ~SqliteCR() noexcept = default;

    //! 同步后查询
    //！线程分离 由引用控制退出
    static void quest_find(SqliteCR *&sq_op);

    //! 同步插入 交换引用的数据
    //! 主动调用 
    static void quest_insert(Alias::_cstr& q, Alias::_cstr& a);

    //! 手动申请实例
    //! init时前后一定时差 存在隐患
    static SqliteCR *getSQCR();
};

#endif // SQLCON

