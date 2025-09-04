#ifndef COMPILE
#define COMPILE
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>  
#include <signal.h> 

//! qml
#include "qtfilewirte.h"
//! py接口
#include "embed.h"
//! 数据结构定义
#include "strpackets.h"
//! 计算对象
#include "runpackets.h"

#define SEM_CREATE     O_CREAT | O_EXCL
#define SEM_GET        O_RDWR
#define SEM_PERMIS     0666
#define NOR_PROCS      5

#ifndef PROCESS
#define PROCESS

//! sql查询对象 静态分配
extern SqliteCR *sq_op;

//! 定义进程函数
template<uint PROCESSES>
class Run
{
protected:
    //! 信号量
    Alias::Sems::_vecsem sems;

    //! 信号量名称
    using _sem_tuple = std::tuple<int, Alias::_cstr>;
    using _vec_tuple = std::vector<_sem_tuple>;
    //! tuple组合 0标识未被子进程占用
    _vec_tuple sems_name;

public:
    //! 初始化所有信号量
    Run();
    //! 销毁所有信号量
    virtual ~Run();

    using _cstr = Alias::_cstr;

    //! 执行函数器 待绑定的函数指针
    using _func = std::function<void()>;
    using _bind_func = void (Run::*)(_cstr);

    //! qml界面
    void QMLgui(_cstr sem_name); //弃用
    //! 推理进程函数
    void RTrunning(_cstr sem_name);

    //! 推理进程函数
    void VMrunning(_cstr sem_name);

    //! 混合节点推理进程函数
    void FXrunning(_cstr sem_name);

    //! sql查找进程
    void SQfinding(_cstr sem_name);

protected:
    //! fixrun
    FixRun<NUMTHS, NUMRT, NUMVM> *fixrun_;
    //! sq
    //sq_op
    //! csvwirte
    csvwirte *pycsv;
};

//! 定义双计算进程
//! 绑定进程接口并执行
template<uint PROCESSES>
class Cess: public Run<PROCESSES>
{
protected:
    //! 执行的进程号
    Alias::Sems::_vecpid pids;

    //! 父进程持有共享内存对象
    PrcsCache *share;

    //! 分词器接口对象
    //! 由父进程持有 
    INdex *pyindex;

public:
    //! 构造所有成员
    Cess();
    //! 销毁成员并kill进程
    ~Cess();

    using typename Run<PROCESSES>::_func;
    using typename Run<PROCESSES>::_bind_func;

    //! 封装函数
    _func getfunc(_bind_func runfunc);

    //! 分配进程执行
    void funcfork(_func run);

    //! 阻塞父进程
    void exec();
    //! 等待所有子进程退出
    void blockwait();
};

#endif //PROCESS

#endif //COMPILE
