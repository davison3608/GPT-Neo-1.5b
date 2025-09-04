#include "runpackets.h"
#include "embed.h"

using namespace Alias;
using namespace Alias::RT;
using namespace std::chrono;
using std::thread;
using std::cout;
using std::cerr;
using namespace std::this_thread;
using namespace std::chrono;

/* baseFIXRun 混合节点计算对象 */
template<>
void baseFIXRun<NUMRT, NUMVM>::Shareanydata()
{
    using namespace std::this_thread;
    cout<<"Assist Threading share pushing is using ...\n";
    bool n;
    //根据退出信号
    while (!this->isdown.load()) {
    //跨进程访问 隐式同步
    this->share->Status(n);
    //无缓存 返回
    //对象进程隐式同步
    if (!n) {
        yield();
        continue;
    }
    //写入到本地队列
    else {
    //临时数据
    int k;
    _cstr quest;
    _vec ids;
    _vec mask;
    //从缓存中取出 内部深拷贝
    this->share->pop_quest(
        k, quest,
        ids, mask
    );
    //引入到队列 
    //内部swap 并隐式同步
    this->que->insert(
        k, quest,
        ids, mask
    );

    yield();
    continue;
    }
    //实际不会执行到这里
    }

    cout<<"baseFIXRun对象 Shareanydata函数结束 !\n";
    return ;
}

template<>
void baseFIXRun<NUMRT, NUMVM>::Queanydata()
{
    using namespace std::this_thread;
    cout<<"Assist Threading queue monitor is using ...\n";
    //根据退出信号
    while (!this->isdown.load()) {
    //锁住计算锁
    _lock lock(this->mt_run);

    //每次查看队列首
    if (this->que->front_get() == nullptr) {
    lock.unlock();

    yield();
    continue;
    }
    //当队列首有数据
    else {
    //仅当空闲节点存在 原子访问
    while (!this->isnode.load()) {
        yield(); //让出时间 其他线程检测
        continue;
    }
    //通知随即计算线程
    lock.unlock();
    this->cv_run.notify_one();

    //让出更多时间保证isnode准确地被更改
    sleep_for(milliseconds(1500));
    continue;
    }
    //不会执行到这里
    }

    cout<<"baseFIXRun对象 Queanydata函数结束 !\n";
    return ;
}

template<>
void baseFIXRun<NUMRT, NUMVM>::Nodeanyfree()
{
    using namespace std::this_thread;
    cout<<"Assist Threading freenode monitor is using ...\n";
    //根据退出信号
    while (!this->isdown.load()) {
    //找到空闲节点重新检测
    re_mon: ;
    //冗余判断
    if (this->isdown.load())
        break;

    int i=0;
    //无需锁 独立更改原子变量
    for ( ; i<NUMRT; i++) {
    //找到rt空闲节点
    if (this->rt_no[i].isfree.load()) {
        this->isnode.store(true); //原子写入
        goto re_mon;
    }
    else
        continue;
    }
    i=0;
    for ( ; i<NUMVM; i++) {
    //找到vm空闲节点
    if (this->vm_no[i].isfree.load()) {
        this->isnode.store(true); //原子写入
        goto re_mon;
    }
    else
        continue;
    }
    //说明无空闲节点
    this->isnode.store(false);
    
    sleep_for(milliseconds(300)); //快速响应
    continue;
    }

    cout<<"baseFIXRun对象 Nodeanyfree函数结束 !\n";
    return ;
}

template<>
baseFIXRun<NUMRT, NUMVM>::baseFIXRun():
RTload(), Soload()
{
    cout<<"\nbaseFIXRun class creating ! just waiting !\n";
    //share 
    this->share=PrcsCache::get();
    //que
    this->que=std::make_unique<Que>();
    //lru
    this->lru=std::make_unique<LRUC<NOR_HASHNODE, NOR_CAPACITY>>();
    //sql
    this->sq_op=SqliteCR::getSQCR();
    cout<<"Struct Data share que lru sql already !\n";
    sleep_for(milliseconds(300));

    //所有推理节点
    for (int i=0; i<NUMRT; i++)
    this->rt_no[i].init();

    for (int i=0; i<NUMVM; i++)
    this->vm_no[i].init();

    //表示运行
    this->isdown.store(false);
    //暂无可推理节点 交由Nodeanyfree处理
    this->isnode.store(false);

    using func = std::function<void()>;

    //detach thread share
    auto s_func_p=&baseFIXRun<NUMRT, NUMVM>::Shareanydata;
    func s_fc=std::bind(s_func_p, this);

    this->t_share=thread(s_fc);
    if (this->t_share.joinable())
        this->t_share.detach();
    
    //detach thread Queanydata
    auto q_func_p=&baseFIXRun<NUMRT, NUMVM>::Queanydata;
    func q_fc=std::bind(q_func_p, this);

    this->t_que=thread(q_fc);
    if (this->t_que.joinable())
        this->t_que.detach();

    //detach thread node
    auto n_func_p=&baseFIXRun<NUMRT, NUMVM>::Nodeanyfree;
    func n_fc=std::bind(n_func_p, this);

    this->t_node=thread(n_fc);
    if (this->t_node.joinable())
        this->t_node.detach();
}

template<>
baseFIXRun<NUMRT, NUMVM>::~baseFIXRun()
{
    //sql
    this->sq_op->destroy();
    delete this->sq_op;

    //所有推理节点
    for (int i=0; i<NUMRT; i++)
    this->rt_no[i].destroy();

    for (int i=0; i<NUMVM; i++)
    this->vm_no[i].destroy();

    //que
    this->que.reset();

    //lru
    this->lru.reset();
}

/* FixRun 实际推理对象 */
template<>
void FixRun<NUMTHS, NUMRT, NUMVM>::rundow()
{
    //原子写入
    this->isdown.store(true);
    //通知所有计算线程
    this->cv_run.notify_all();
    return ;
}

template<>
void FixRun<NUMTHS, NUMRT, NUMVM>::ContxSar()
{
    using namespace std::this_thread;
    sleep_for(milliseconds(500));

    while (!this->isdown.load()) {
    //总是等待被唤醒
    _lock lock(this->mt_run);
    this->cv_run.wait(lock);
    cout<<"Inference Thread is noticed !\n";
    cout<<"Looking for Inference Node !\n";

    //检查是否停止
    if (this->isdown.load())
        break;

    rtNode *exec_rtnode=nullptr;
    vmNode *exec_vmnode=nullptr;
    //被通知que队首 且此时有空闲节点
    for (int i=0; i<NUMRT; i++) {
    //找到空闲节点 必须立马变更状态
    if (this->rt_no[i].isfree.load()) {
        rt_no[i].isfree.store(false);
        exec_rtnode=&rt_no[i];
        goto find_free;
    }
    else 
        continue;
    }
    for (int i=0; i<NUMVM; i++) {
    if (this->vm_no[i].isfree.load()) {
        vm_no[i].isfree.store(false);
        exec_vmnode=&vm_no[i];
        goto find_free;
    }
    else 
        continue;
    }

    //找到空闲节点直接跳转
    find_free: ;
    //取出队首 隐式同步 
    queNode deal_node; //swap到临时节点
    this->que->front_release(deal_node); //同时内部释放首节点

    //已更改que 解锁后异步推理
    lock.unlock();

    using _fixRT = void(FixRun<NUMTHS, NUMRT, NUMVM>::*)(rtNode *&, queNode&);
    using _fixVM = void(FixRun<NUMTHS, NUMRT, NUMVM>::*)(vmNode *&, queNode&);
    using _exec = std::function<void()>;

    if (exec_rtnode != nullptr) {
    //包装后执行 计算函数引用到临时变量
    _fixRT func_rt=&FixRun<NUMTHS, NUMRT, NUMVM>::RTInference;
    _exec run_rt=std::bind(
        func_rt, this, 
        std::ref(exec_rtnode), std::ref(deal_node)
    );

    run_rt(); //执行
    exec_rtnode->isfree.store(true); //结束标记空闲
    continue; //重新等待被通知
    }
    else if (exec_vmnode != nullptr) {
    _fixVM func_vm=&FixRun<NUMTHS, NUMRT, NUMVM>::VMInference;
    _exec run_vm=std::bind(
        func_vm, this,
        std::ref(exec_vmnode), std::ref(deal_node)
    );

    run_vm();
    exec_vmnode->isfree.store(true);
    continue;
    }
    //实际不会执行到这里
    }

    cout<<"FIXRun对象 ContxSar函数结束 !\n";
    return ;
}

template<>
FixRun<NUMTHS, NUMRT, NUMVM>::FixRun():
baseFIXRun<NUMRT, NUMVM>()
{
    using _run = std::function<void()>;
    sleep_for(milliseconds(300));
    cout<<"\nFIXRun class creating ! just waiting !\n";
    
    //lru thread
    auto run_lru_p=&FixRun<NUMTHS, NUMRT, NUMVM>::LRUCQuery;
    _run run_lru=std::bind(run_lru_p, this);

    _thread run_lru_t(run_lru);
    if (run_lru_t.joinable())
        run_lru_t.detach();
    
    //脚本加载
    this->pydeser=std::make_unique<deser>();

    sleep_for(milliseconds(300));
    //线程序列
    this->threads.resize(NUMTHS);
    uint count_t=0;

    //detach thread   
    //0.5后并发计算线程
    std::this_thread::sleep_for(milliseconds(500));
    for (auto&e : this->threads) {
    auto run_p=&FixRun<NUMTHS, NUMRT, NUMVM>::ContxSar;
    _run run=std::bind(run_p, this);

    e=_thread(run);

    if (e.joinable()) {
        cout<<"Inference Threading "<<count_t;
        cout<<" Successfully Starting ...\n";
        e.detach();
    }

    count_t++;
    }
    cout<<"Inference Threading All concurrently over ...\n";
    std::this_thread::sleep_for(milliseconds(500));
    cout<<"FIXRun to FIXRun aleady created !\n";
}

template<>
FixRun<NUMTHS, NUMRT, NUMVM>::~FixRun()
{ 
    this->pydeser.reset();
    //thread no need to wait
    //return
}
