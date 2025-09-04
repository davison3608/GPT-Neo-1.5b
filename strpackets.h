#ifndef C_INCLUDE
#define C_INCLUDE
#include "iostream"
#include "assert.h"
#include "cstring"
#include "vector"
#include "atomic"
#include "tuple"

#include "functional"
#include "thread"
#include "mutex"
#include "condition_variable"

#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <stdio.h>
#include <stdlib.h>
#include <semaphore.h>
#endif //C_INCLUDE

#ifndef SHARE_CA
#define SHARE_CA
#include <cuda.h>
#include <cuda_runtime.h>

//! 通用类型别名
namespace Alias {
    using _cstr = std::string;
    using _usize = std::size_t;
    using _size = ssize_t;

    using precison = int64_t;
    using _vecus = std::vector<_usize>;
    using _vec = std::vector<precison>;
    using _nor_tuple = std::tuple<_usize, _cstr>;

    using _mtx = std::mutex;
    using _pmtx = pthread_mutex_t;
    using _pmtx_attr = pthread_mutexattr_t;

    using _lock = std::unique_lock<_mtx>;
    using _cv = std::condition_variable;
    using _flag = std::atomic<bool>;

namespace Sems {
    using _pid = pid_t;
    using _vecpid = std::vector<_pid>;

    using _sem = sem_t*;
    using _vecsem = std::vector<_sem>;
} // namespace Sems

namespace RT {
    using _cus = cudaStream_t;
    using _cue = cudaEvent_t;
} // namespace RT

} // namespace TypeAlias

#define SHE_CREATE      IPC_CREAT | 0666 //内存创建
#define SHE_GET         0666 //内存id获取
#define SHE_REMOVE      IPC_RMID //内存释放
#define MAX_QUEST       50 //问题索引化长度限定
#define MAX_QUEST_STR   MAX_QUEST + MAX_QUEST / 2 //问题char长度   

//! 共享内存缓冲区
//! 保证多进程访问安全
class PrcsCache
{
private:
    //! 对话查找key 问题数据
    Alias::_usize cache_key = 0;
    char cache_quest[MAX_QUEST_STR] = {};
    
    //! 索引 掩码
    Alias::precison ids[MAX_QUEST] = {};
    Alias::precison mask[MAX_QUEST] = {};
    
    //! 进程锁
    Alias::_pmtx pmtx;
    //! 引用计数 为0时销毁
    Alias::_usize counts;

    bool iscache; //! 标志当前缓存有效
    Alias::_usize id; //! 内存id

public:
    PrcsCache() noexcept = default;
    ~PrcsCache() noexcept = default;

    //! 手动构造析构 不保证多进程安全
    static PrcsCache *init();
    void destory();

    //! 返回对象指针 
    static PrcsCache *get();

    //! 返回缓存状态
    void Status(bool& n);

    //! 写入缓存 深拷贝副本
    void push_quest(
        int& key, Alias::_cstr& question, 
        Alias::precison *ids, 
        Alias::precison *mask,
        Alias::_usize seq
    );
    //! 取出缓存 深拷贝副本
    void pop_quest(
        int& key, Alias::_cstr& question, 
        Alias::_vec& ids, 
        Alias::_vec& mask
    );
};

#ifndef QUEUE
#define QUEUE
//! 队列节点
struct queNode {
    Alias::_usize key;    
    Alias::_cstr quest;
    Alias::_vec ids; //MAX_QUEST
    Alias::_vec mask; //MAX_QUEST
    queNode *next; //指向下一个节点

    queNode(): queNode(0, "")
    {}
    queNode(const Alias::_usize& k, const Alias::_cstr& q):
    key(k), quest(q), next(nullptr)
    {}
};
//! 待处理任务队列 保证多线程访问安全
//! 每个进程都拥有
class Que
{
private:
    //! 指向首尾节点
    queNode *first; //next指向实际首节点
    queNode *last; //总是指向实际的尾节点
    //! 有效节点数量
    Alias::_usize counts;
    //! 锁
    Alias::_mtx mt_que;
public:
    Que() noexcept; 
    ~Que() noexcept;

    //! 从尾插入节点 深拷贝待推理数据
    void insert(
        int& key, Alias::_cstr& quest, 
        Alias::_vec& ids, Alias::_vec& mask
    );

    //! 获取队列首节点指针
    queNode* front_get() noexcept;

    //! 释放队列首节点
    //! 同时拷贝到引用节点 保证多线程访问安全
    void front_release(queNode& or_node);
};

#endif //QUEUE

#include "set"
#define NOR_HASHNODE    1000 //桶容量        
#define NOR_CAPACITY    10 //缓存容量

#ifndef HASHLRU
#define HASHLRU
//! 表节点
struct hashNode {
    Alias::_usize key;
    Alias::_cstr quest;
    Alias::_cstr answer;
    hashNode *next; //表中指向下一个节点

    hashNode *left; //用于list 指向左右节点
    hashNode *right;

    hashNode(): hashNode(0, "", "")
    {}
    hashNode(const Alias::_usize& k, const Alias::_cstr& q, const Alias::_cstr& s):
    key(k), quest(q), answer(s), next(nullptr), left(nullptr), right(nullptr)
    {}
    hashNode(const hashNode& _or)
    {
    this->key=_or.key;
    this->quest=_or.quest;
    this->answer=_or.answer;

    this->next=_or.next;
    this->left=_or.left;
    this->right=_or.right;
    }
};
//! LRU缓存 保证多线程访问安全
//! RT进程独有
template<uint NODES, uint LRUNUM>
class LRUC
{
private:
    //! 锁
    Alias::_mtx mt_hs;
    //! 当前有效节点数量 允许节点容量 
    Alias::_usize counts;
    Alias::_usize capacity;

    //! 索引映射
    Alias::_usize hashfunction(Alias::_usize k) const noexcept;

    //! 是否缓存溢出
    bool isfull() const noexcept;
    
    //! 删除链尾 
    void remove();

    //! 节点更换到list首部 提升优先级
    void refresh(hashNode *node); 

public:
    LRUC() noexcept;
    ~LRUC() noexcept;

    using _vecnode_p = std::vector<hashNode*>;
    using _set = std::set<Alias::_usize>;

    //! 根据数据插入新节点
    hashNode *insert(Alias::_usize& k, Alias::_cstr& qu, Alias::_cstr& as);

    //! 查找值 没有则返回空
    hashNode *find(Alias::_usize& k);

    //! 打印当前缓存的所有key
    void getkeys() noexcept;

private:
    //! 哈希桶
    _vecnode_p Nodes;

    //! 节点链表首尾 构造时指向边界
    hashNode *first;
    hashNode *last;
    //! 保存有效节点key 独一性
    _set keys;
};

#endif //HASHLRU

#include <fcntl.h>
#include <fstream> 
#endif //SHARE_CA
