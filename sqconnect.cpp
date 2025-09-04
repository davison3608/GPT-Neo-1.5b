#include "sqconnect.h"
#include <future> 

using std::cout;
using std::cerr;
using namespace Alias;
using namespace Alias::SQ;

_pmtx pmt_sq;
_pmtx_attr pmt_sq_attr;
_usize sq_ref = 0;

static void init_pmt()
{
    //进程间共享
    pthread_mutexattr_init(&pmt_sq_attr);
    pthread_mutexattr_setpshared(&pmt_sq_attr, PTHREAD_PROCESS_SHARED);

    if (pthread_mutex_init(&pmt_sq, &pmt_sq_attr) == -1)
    { cerr<<"sql进程间锁属性创建错误 强制停止\n"; exit(1); }

    return ;
}

static void setsqdata_time(Sqdata& data) 
{
    using namespace std::chrono;
    using namespace std;
    auto now=system_clock::now();
    auto time=system_clock::to_time_t(now);

    auto time_t=localtime(&time);
    auto year=1900 + time_t->tm_year;
    auto month=1 + time_t->tm_mon;
    auto day=time_t->tm_mday;

    data.year=to_string(year);
    data.month=to_string(month);
    data.day=to_string(day);
}

static bool sql_insert(Sqdata& data)
{
    if (conn_ == nullptr) {
        cerr<<"错误：数据库连接已断开（conn_ 为空）\n";
        return false;
    }

    //SQL模板
    const char* sql="INSERT INTO your_table (col1, col2, col3, col4, col5) VALUES (?, ?, ?, ?, ?)";

    const char* check_table_sql = "CREATE TABLE IF NOT EXISTS your_table ("
        "id INT AUTO_INCREMENT PRIMARY KEY,"
        "col1 VARCHAR(255) NOT NULL,"
        "col2 VARCHAR(255) NOT NULL,"
        "col3 VARCHAR(20) NOT NULL,"
        "col4 VARCHAR(20) NOT NULL,"
        "col5 VARCHAR(50) NOT NULL)"; 
    
    // 执行建表语句
    if (mysql_query(conn_, check_table_sql) != 0) {
        cerr << "创建表失败: " << mysql_error(conn_) << '\n';
        return false;
    }

    // 存在则清空表中已有数据
    const char* truncate_sql = "TRUNCATE TABLE your_table";
    if (mysql_query(conn_, truncate_sql) != 0) {
        cerr << "清空表失败: " << mysql_error(conn_) << '\n';
        return false;
    }

    //预处理语句句柄
    MYSQL_STMT* stmt=mysql_stmt_init(conn_);
    if (stmt == nullptr) {
    cerr<<"预处理句柄初始化失败 " << mysql_error(conn_)<<'\n';
    return false;
    }
    
    //sql语句
    if (mysql_stmt_prepare(stmt, sql, strlen(sql)) != 0) {
    cerr<<"准备语句失败: "<<mysql_stmt_error(stmt)<<'\n';
    mysql_stmt_close(stmt);
    return false;
    }

    //绑定参数
    MYSQL_BIND params[5];
    memset(params, 0, sizeof(params));
    
    params[0].buffer_type=MYSQL_TYPE_STRING;
    params[0].buffer=(char*)data.quest.c_str();
    params[0].buffer_length=data.quest.size();
    
    params[1].buffer_type=MYSQL_TYPE_STRING;
    params[1].buffer=(char*)data.answer.c_str();
    params[1].buffer_length=data.answer.size();
    
    params[2].buffer_type=MYSQL_TYPE_STRING;
    params[2].buffer=(char*)data.year.c_str();
    params[2].buffer_length=data.year.size();
    
    params[3].buffer_type=MYSQL_TYPE_STRING;
    params[3].buffer=(char*)data.month.c_str();
    params[3].buffer_length=data.month.size();

    params[4].buffer_type=MYSQL_TYPE_STRING;
    params[4].buffer=(char*)data.day.c_str();
    params[4].buffer_length=data.day.size();
    
    //执行绑定
    if (mysql_stmt_bind_param(stmt, params) != 0) {
    cerr<<"绑定失败: "<<mysql_stmt_error(stmt)<<'\n';
    mysql_stmt_close(stmt);
    return false;
    }
    //执行插入
    if (mysql_stmt_execute(stmt) != 0) {
    cerr<<"插入失败: "<<mysql_stmt_error(stmt)<<'\n';
    mysql_stmt_close(stmt);
    return false;
    }

    mysql_stmt_close(stmt);
    return true;
}

const _info host_ = "localhost";
const _info user_ = "davison";
const _info pass_ = "202069";
_info sql_name = "talk_sql";

_sql *conn_;
_port port_ = 3306;

void Sqlite::init_connect()
{
    //保证一次连接
    if (sq_ref > 0)
    return ;

    cout<<"sql try to connected ...\n";
    //全局锁构造
    init_pmt();

    //初始化句柄
    conn_=mysql_init(nullptr);
    assert(conn_);

    //连接
    auto status=mysql_real_connect(
        conn_, 
        host_.c_str(),
        user_.c_str(),
        pass_.c_str(),
        sql_name.c_str(),
        port_,
        nullptr, 0
    );
    if (status == nullptr) {
    cerr<<"sql failed "<<mysql_error(conn_)<<'\n';
    cerr<<"errno "<<mysql_errno(conn_)<<'\n';
    exit(1);
    }

    //测试
    Sqdata test;
    test.quest="sql connect test";
    test.answer="test is pass ...";
    setsqdata_time(test);
    if (!sql_insert(test))
    { cerr<<"sql 测试失败 已返回\n"; return ; }
    cout<<"sql is conntected and test is passed ...\n";
    cout<<"sql_info >> host: "<<host_<<" user: "<<user_<<" sql_db: "<<sql_name;
    cout<<std::endl<<std::endl;

    //全局引用递增
    pthread_mutex_lock(&pmt_sq);
    sq_ref++;
    pthread_mutex_unlock(&pmt_sq);
    return ;
}

void Sqlite::destroy()
{
    //锁住
    pthread_mutex_lock(&pmt_sq);
    
    //引用递减
    if (sq_ref > 0) {
    sq_ref--;
    pthread_mutex_unlock(&pmt_sq);
    return ;
    }
    //最后一次释放
    else {
    pthread_mutex_unlock(&pmt_sq);
    
    //解锁后断开连接
    mysql_close(conn_);
    //销毁锁
    pthread_mutexattr_destroy(&pmt_sq_attr);
    pthread_mutex_destroy(&pmt_sq);
    }
    return ;
}

/* SqliteCR sql增查部分 */
static void sql_find_async(SqliteCR *&sq_op)
{
    //test
    return ;
    //单独窗口用于sql query
    //连接到qml
    while (sq_ref > 0) {
    cout<<"cin find to query sql\n";
    cout<<"cin break to down thread\n";
    
    cout<<"your cmd: ";
    _cstr cmd="";
    std::getline(std::cin, cmd);
    
    //break表示退出
    if (strcmp(cmd.c_str(), "break") == 0)
    break;

    //find表示查询
    if (strcmp(cmd.c_str(), "find") == 0) {
    //锁住 
    //一次查询可能使跨进程写入等待
    pthread_mutex_lock(&pmt_sq);

    //查询所有行并打印
    //
    //
    //
    //
    //
    //结束后解锁
    pthread_mutex_unlock(&pmt_sq);
    continue;
    }

    //未知
    else
    cerr<<"\nunkwon cmd and continue\n";
    }

    return ;
}

void SqliteCR::quest_find(SqliteCR *&sq_op)
{
    using _fur = std::future<void>;
    cout<<"sql find thread is using ...\n";

    _fur asy_fur=std::async(
        std::launch::deferred, sql_find_async,
        std::ref(sq_op)
    );
    //启动线程 阻塞等待
    asy_fur.wait();

    cout<<"sql查询线程已退出\n";
    return ;
}

void SqliteCR::quest_insert(_cstr& q, _cstr& a)
{
    //临时对象
    Sqdata data;
    data.quest.swap(q);
    data.answer.swap(a);
    //设置插入日期
    setsqdata_time(data);
    
    //锁住
    pthread_mutex_lock(&pmt_sq);
    //插入
    if (!sql_insert(data))
    cerr<<"sql插入失败 忽略本次数据 question is "<<data.quest<<'\n';
    cout<<"对话已记录 question is "<<data.quest<<'\n';

    //完成后解锁
    pthread_mutex_unlock(&pmt_sq);
    return ;
}

SqliteCR *SqliteCR::getSQCR()
{
    //手动获取一个可执行接口
    auto new_ptr=new SqliteCR();
    
    //需要在外部destroy
    if (sq_ref > 0) {
    //引用计数递增
    sq_ref++;
    return new_ptr;
    }
    
    //第一次则初始化全局变量
    SqliteCR::init_connect();
    return new_ptr;
}
