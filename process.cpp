#include "process.h"
#include <sys/wait.h>
#include <unistd.h>

using std::cout;
using std::cerr;
using namespace Alias;
using namespace Alias::Sems;

SqliteCR *sq_op;

/* Run 定义进程函数 */
template<>
Run<NOR_PROCS>::Run() 
{
    cout<<"Process class creating !\n";
    this->sems.resize(NOR_PROCS);
    this->sems_name.resize(NOR_PROCS);
    Alias::_cstr basename="/runfuncsem";
    int counts=1;

    //初始化信号量
    cout<<"process sems try to get and set ...\n";
    for (int i=0; i<NOR_PROCS; i++) {
    Alias::_cstr name=basename + std::to_string(counts);
    //名称标识0说明没有绑定到进程函数
    this->sems_name[i]=std::make_tuple(0, name);
    
    //保险先清除信号量
    sem_unlink(name.c_str());

    //初始化0 后续wait会阻塞
    this->sems[i]=sem_open(name.c_str(), SEM_CREATE, SEM_PERMIS, 0); 
    if (this->sems[i] == SEM_FAILED) {
        cerr<<"\n进程信号量创建错误 强行停止 ";
        cerr<<strerror(errno)<<"\n"; //错误信息
        sem_unlink(name.c_str());
        exit(1);
    } 

    counts++;
    }
    cout<<"process sems finish ...";
    cout<<std::endl<<std::endl;
    //仅指向空
    this->fixrun_ = nullptr;
    sq_op = nullptr;
    this->pycsv = nullptr;
}

template<>
Run<NOR_PROCS>::~Run()
{ 
    for (auto& e: this->sems) {
    //分离信号量
    sem_close(e);
    }
    for (auto& e: this->sems_name) {
    auto& name=std::get<1>(e);
    sem_unlink(name.data());
    }
}

/* Cess 定义计算进程 */
template<>
Cess<NOR_PROCS>::Cess()
{ 
    //预留进程序列
    this->pids.reserve(NOR_PROCS); 
    //构造share对象
    this->share=PrcsCache::init(); 
      
    //分配sql接口
    sq_op=SqliteCR::getSQCR(); 
    //脚本接口
    this->pyindex=new INdex();
    this->pycsv=new csvwirte();
}

template<>
Cess<NOR_PROCS>::~Cess()
{ 
    for (auto& e: this->pids) {
    if (e != -1)
        ::kill(e, SIGKILL);
    }
    //销毁接口
    delete this->pyindex;
    delete this->pycsv;
    //手动销毁share对象
    //由子进程执行
    this->share->destory();
    //引用递减 销毁sql接口对象
    sq_op->destroy();
    delete sq_op;
}

template<>
Run<NOR_PROCS>::_func Cess<NOR_PROCS>::getfunc(Run<NOR_PROCS>::_bind_func runfunc)
{
    //获取没有绑定的信号量名称
    _cstr name;
    for (auto& e: this->sems_name) {
    if (std::get<0>(e) == 0) {
        name=std::get<1>(e);
        //表示已绑定
        std::get<0>(e)=1;
        //分配后退出
        break;
    }
    else
        continue;
    }
    //绑定 并分配信号
    _func func=std::bind(runfunc, this, name);
    //cout<<"进程函数分配信号 "<<name<<std::endl; //debug
    //返回 函数器重载=
    return func;
}

template<>
void Cess<NOR_PROCS>::funcfork(Run<NOR_PROCS>::_func run)
{
    //分配进程
    _pid i=::fork();
    //分别操作
    if (i == -1)
    { cerr<<"进程创建错误 已返回\n"; return ; }
    //子进程执行
    if (i == 0) {
        run();
    }
    else {
    this->pids.push_back(i);
    }
    //返回 非阻塞
    return ;
}

template<>
void Cess<NOR_PROCS>::exec()
{
    cout<<"\n父进程已阻塞 down命令退出\n";
    cout<<"request your question: ";
    Alias::_cstr down="down";

    using namespace std::this_thread;
    //循环等待
    while (1) {
    //提前声明待输入
    Alias::_cstr cmd="";
    Alias::_cstr k="";

    std::getline(std::cin, cmd);
    if (strcmp(down.c_str(), cmd.c_str()) == 0) {
    //所有信号量通知
    for (int i=0; i<NOR_PROCS; i++) {
    if (std::get<0>(this->sems_name[i]) == 1) {
        //对应位置信号量
        auto sem_p=this->sems[i];
        //通知
        if (sem_post(sem_p) == -1)
            cerr<<"父进程通知信号量失败 \n";
        //分离已通知的信号
        sem_close(sem_p);
    }
    }
    break;
    }
    //对于问题 转到share缓冲区
    else {
    cout<<"set key(0~999): ";
    std::getline(std::cin, k);
    int key=atoi(k.data());

    if (key < 0 || key >= 1000) { 
        cerr<<"\n设置的key在0~999 重新输入\n"; 
        goto set_continue; 
    }
    
    //父进程调用
    //一次性分词写入share this->pyindex->getindex(cmd); 弃用

    //获取序列指针
    precison *ids_data=this->pyindex->getids_ptr();
    precison *mask_data=this->pyindex->getmask_ptr();
    auto seq=this->pyindex->getsequence();
    //写入缓存 
    //内部隐式同步
    this->share->push_quest(
        key, cmd, 
        ids_data, mask_data,
        seq
    );
    //结束本次写入
    }
    cout<<"\nquestion已写入缓存 key = "<<k;

    set_continue:;
    cout<<"request your question: ";
    continue;
    }

    return ;
}

template<>
void Cess<NOR_PROCS>::blockwait()
{
    //阻塞等待
    for (auto&e : this->pids) {
        waitpid(e, nullptr, 0);
    }
    cout<<"父进程已退出 pid "<<::getpid()<<"\n";
    //父进程退出
    cout<<"所有子进程退出 Cess析构 !\n";
}
