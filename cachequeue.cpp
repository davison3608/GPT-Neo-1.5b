#include "strpackets.h"
#include <algorithm>

using namespace Alias;
using std::cout;
using std::cerr;
using std::make_tuple;
using std::copy_n;
using namespace std::this_thread;
/* PrcsCache 共享内存缓冲区 */
PrcsCache *PrcsCache::init()
{
    cout<<"process share memory try to create ...\n";
    key_t gen=::ftok("../tmp", 88);
    if (gen < 0) 
    { cerr<<"key生成错误 强制停止\n"; exit(1); }

    _usize size=sizeof(PrcsCache);
    auto shm_id=::shmget(gen, size, SHE_CREATE);
    if (shm_id == -1) { 
        cerr<<"共享内存对象创建错误 强制停止 ";
        cerr<<strerror(errno)<<" 错误码 " <<errno; 
        exit(1); 
    }
    
    //获取内存映射
    auto _p=::shmat(shm_id, nullptr, 0);
    assert(_p);
    auto share=static_cast<PrcsCache*>(_p);
    share->id=shm_id;
    
    //初始化成员
    _pmtx_attr attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);

    if (pthread_mutex_init(&share->pmtx, &attr) == -1)
    { cerr<<"进程间锁属性创建错误 强制停止\n"; exit(1); }
    
    share->cache_key=0;
    memset(share->cache_quest, 0, sizeof(cache_quest));
    share->iscache=false;
    
    share->counts=0;
    cout<<"share process memory aleady created ...";
    cout<<std::endl<<std::endl;
    return share;
}

void PrcsCache::destory()
{
    pthread_mutex_lock(&this->pmtx);
    //引用次数递减
    if (this->counts > 0) { 
    this->counts--; 
    pthread_mutex_unlock(&this->pmtx);
    ::shmdt(this); 
    return ; 
    }
    //引用计数0销毁
    else if (this->counts == 0) {
    pthread_mutex_unlock(&this->pmtx);
    pthread_mutex_destroy(&this->pmtx);

    this->iscache=false;
    ::shmctl(this->id, SHE_REMOVE, nullptr);

    ::shmdt(this);
    return ;
    }
}

PrcsCache *PrcsCache::get()
{
    key_t gen=::ftok("../tmp", 88);
    if (gen < 0) 
    { cerr<<"key生成错误 强制停止\n"; exit(1); }

    _usize size=sizeof(PrcsCache);
    auto shm_id=::shmget(gen, size, SHE_GET);
    if (shm_id == -1) 
    { cerr<<"共享内存对象获取错误 强制停止\n"; exit(1); }
    
    //获取内存映射
    auto _p=::shmat(shm_id, nullptr, 0);
    assert(_p);
    auto share=static_cast<PrcsCache*>(_p);
    assert(share->id==shm_id);

    //锁住
    pthread_mutex_lock(&share->pmtx);
    //引用次数自增后返回
    share->counts++;
    pthread_mutex_unlock(&share->pmtx);
    return share;
}

void PrcsCache::Status(bool& n)
{
    //锁住
    pthread_mutex_lock(&this->pmtx);
    n=this->iscache;

    pthread_mutex_unlock(&this->pmtx);
    return ;
}

void PrcsCache::push_quest(int& key, _cstr& question, precison *ids, precison *mask, _usize seq)
{
    //锁住
    pthread_mutex_lock(&this->pmtx);

    //覆盖key string缓存
    this->cache_key=key;
    _usize copy_len=std::min(question.length(), static_cast<_usize>(MAX_QUEST_STR - 1));
    strncpy(
        this->cache_quest, 
        question.data(),
        copy_len
    );
    this->cache_quest[copy_len]='\0';

    //覆盖容器 只覆盖到seq长度
    for (int i=0; i<seq; i++) {
        this->ids[i]=ids[i];
        this->mask[i]=mask[i];
    }
    this->iscache=true;

    pthread_mutex_unlock(&this->pmtx);
    return ;
}

void PrcsCache::pop_quest(int& key, _cstr& question, _vec& ids, _vec& mask)
{
    ids.resize(MAX_QUEST);
    mask.resize(MAX_QUEST);

    //锁住
    pthread_mutex_lock(&this->pmtx);
    //缓存无效或来不及命中
    if (this->iscache)
    return ;

    //取出
    key=this->cache_key;
    question.clear();
    question=this->cache_quest; //string重载=
    
    //从头覆盖
    for (int i=0; i<MAX_QUEST; i++) {
    ids[i]=this->ids[i];
    mask[i]=this->mask[i];

    this->ids[i]=0; //覆盖后重置
    this->mask[i]=0;
    }
    //重置缓存
    this->cache_key=0;
    memset(this->cache_quest, 0, sizeof(cache_quest));
    this->iscache=false;

    pthread_mutex_unlock(&this->pmtx);
    return ;
}

/* Que 待处理任务队列 */
Que::Que() noexcept:
counts(0), first(new queNode()), last(first) 
{}

Que::~Que()
{
    for (int i=0; i<counts; i++) {
    auto tmp=this->first->next;
    this->first->next=tmp->next;
    delete tmp;
    }
    delete this->first;
}

void Que::insert(int& key, _cstr& quest, _vec& ids, _vec& mask)
{
    _lock lock(this->mt_que, std::defer_lock);
    //构造新节点
    queNode *new_node=new queNode(key, quest);
    //锁住
    lock.lock();

    //last指向实际节点
    this->last->next=new_node;
    this->last=new_node;
    this->counts++;

    //单独处理序列部分 新节点容器长度为MAX_QUEST
    new_node->ids.resize(MAX_QUEST);
    new_node->mask.resize(MAX_QUEST);
    assert(ids.size() == MAX_QUEST);
    assert(mask.size() == MAX_QUEST);
    //交换数据 队列持有
    new_node->ids.swap(ids);
    new_node->mask.swap(mask);

    lock.unlock();
    return ;
}

queNode* Que::front_get() noexcept
{
    //同步后返回
    std::lock_guard<_mtx> lock(this->mt_que);
    return this->first->next;
    //自动解锁
}

void Que::front_release(queNode& or_node) 
{
    _lock lock(this->mt_que, std::defer_lock);
    //锁住
    lock.lock();

    //首无节点则返回
    if (this->first->next == nullptr)
    { lock.unlock(); return ; }

    auto tmp=this->first->next;
    this->first->next=tmp->next;

    //重置引用容器 长度保证一致
    or_node.ids.resize(MAX_QUEST);
    or_node.mask.resize(MAX_QUEST);
    //互换资源 
    or_node.key=tmp->key;
    or_node.quest=tmp->quest;
    or_node.ids.swap(tmp->ids);
    or_node.mask.swap(tmp->mask);

    delete tmp;
    this->counts--;

    //无节点重置last
    if (this->first->next == nullptr) 
    this->last=this->first;

    lock.unlock();
    return ;
}
