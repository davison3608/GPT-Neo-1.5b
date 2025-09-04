#include "strpackets.h"

using namespace Alias;
using namespace std::this_thread;
using std::cout;
using std::cerr;
/* LRUC LRU缓存 */
template<>
LRUC<NOR_HASHNODE, NOR_CAPACITY>::LRUC() noexcept:
counts(0), capacity(NOR_CAPACITY)
{
    this->first=new hashNode();
    this->last=new hashNode(); 
    this->first->right=this->last;
    this->last->left=this->first;
    this->Nodes.resize(NOR_HASHNODE);
}

template<>
LRUC<NOR_HASHNODE, NOR_CAPACITY>::~LRUC()
{
    for (int i=0; i<counts; i++) {
    auto tmp=this->first->right;
    this->first->right=tmp->right;
    delete tmp;
    }

    delete this->first;
    delete this->last;
}

template<>
_usize LRUC<NOR_HASHNODE, NOR_CAPACITY>::hashfunction(_usize k) const noexcept
{
    //增加随机性
    k ^= (k >> 16);
    k ^= (k >> 8);
    k = k % NOR_HASHNODE;
    return k;
}

template<>
bool LRUC<NOR_HASHNODE, NOR_CAPACITY>::isfull() const noexcept
{ return this->capacity <= this->counts; }

template<>
void LRUC<NOR_HASHNODE, NOR_CAPACITY>::remove()
{
    auto tmp=this->last->left;
    auto tmp_k=tmp->key;
    //重连接尾部
    tmp->left->right=this->last;
    this->last->left=tmp->left;
    //表重连
    auto id=this->hashfunction(tmp_k);
    auto curr=this->Nodes[id];
    hashNode *prev;
    
    while (curr != nullptr) {
    if (tmp_k == curr->key) {
        //节点首删除
        if (prev == nullptr)
            this->Nodes[id]=curr->next;
        //节点中间删除
        else
            prev->next=curr->next;
        break;
    }
    prev=curr;
    curr=curr->next;
    }

    delete tmp;
    this->counts--;

    //更新keys
    this->keys.erase(tmp_k);
    return ;
}

template<>
void LRUC<NOR_HASHNODE, NOR_CAPACITY>::refresh(hashNode *node)
{
    //新节点进入list
    if (node->left == nullptr && node->right == nullptr) {
    //从头插入
    auto tmp=this->first->right;
    tmp->left=node;
    node->right=tmp;
    node->left=this->first;
    this->first->right=node;
    
    return ;
    }
    //对于旧节点重排列
    auto node_left=node->left;
    auto node_right=node->right;
    node_left->right=node_right;
    node_right->left=node_left;

    auto tmp=this->first->right;
    tmp->left=node;
    node->right=tmp;
    node->left=this->first;
    this->first->right=node;

    return ;
}

template<>
hashNode *LRUC<NOR_HASHNODE, NOR_CAPACITY>::insert(_usize& k, _cstr& qu, _cstr& as)
{
    _lock lock(this->mt_hs, std::defer_lock);
    auto new_node=new hashNode(k, qu, as);
    
    //获取映射
    auto id=this->hashfunction(k);
    //锁住
    lock.lock();
    //访问对应节点
    auto old_node=this->Nodes[id];
    //从首新插入节点
    if (old_node == nullptr) {
        this->Nodes[id]=new_node;
    }
    
    //如果已存在缓存则更新
    while (old_node != nullptr) {
    if (old_node->key == k) {
        old_node->answer.swap(new_node->answer);
        this->refresh(old_node); //提升优先级
        lock.unlock();
        delete new_node; //无需插入
        return old_node;
    }
    else 
        old_node=old_node->next;
    }

    //从首插入
    auto tmp=this->Nodes[id];
    new_node->next=tmp;
    this->Nodes[id]=new_node;
    this->refresh(new_node); //提升优先级
    this->counts++;

    //溢出则刷新
    if (this->isfull())
    this->remove();

    //新列入keys
    this->keys.insert(k);
    
    lock.unlock();
    return new_node;
}

template<>
hashNode *LRUC<NOR_HASHNODE, NOR_CAPACITY>::find(_usize& k)
{
    _lock lock(this->mt_hs, std::defer_lock);
    //获取映射
    auto id=this->hashfunction(k);
    //同步后查找
    lock.lock();
    auto old_node=this->Nodes[id];
    //若对应节点空
    if (old_node == nullptr) { 
        lock.unlock();
        return nullptr;
    }

    while (old_node != nullptr) {
    if (old_node->key == k) {
        this->refresh(old_node); //提升优先级
        lock.unlock();
        return old_node;
    }
    else 
        old_node=old_node->next;
    }
    //无匹配则返回空
    lock.unlock();
    return nullptr;
}

template<>
void LRUC<NOR_HASHNODE, NOR_CAPACITY>::getkeys() noexcept
{
    //同步后打印key
    std::lock_guard<_mtx> lock(this->mt_hs);
    cout<<"\nlru可查找有效key ";
    for (auto& e: this->keys)
    if (e != 0)
        cout<<e<<" ";
    cout<<'\n';
    return ;
}