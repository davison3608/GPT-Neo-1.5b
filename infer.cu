#include "runpackets.h"
#include "sopackets.h"
#include <iostream>

using namespace Alias;
using std::cerr;
using std::cout;

static void inference_test(rtNode *&node, queNode& tmp)
{
    assert(node);
    cout<<"节点test key "<<tmp.key<<'\n';
    cout<<"节点test question "<<tmp.quest<<'\n';
    cout<<"节点test ids ";
    for (auto& e: tmp.ids)
        cout<<e<<" "; 
    cout<<'\n';
    cout<<"节点test mask ";
    for (auto& e: tmp.mask)
        cout<<e<<" "; 
    cout<<'\n';
}


/* FixRun对象 核心推理函数 */
template<>
void FixRun<NUMTHS, NUMRT, NUMVM>::RTInference(rtNode *&node, queNode& tmp)
{
    cout<<"RT节点被分配 run inference ...\n";

    //单独窗口显示回答
    //连接到qml


    //写入到lru缓存 隐式同步


    //写入sql 跨进程同步

    //test
    inference_test(node, tmp);
    return ;
}

template<>
void FixRun<NUMTHS, NUMRT, NUMVM>::VMInference(vmNode *&node, queNode& tmp)
{
    cout<<"VM节点被分配 run inference ...\n";


    //单独窗口显示回答
    //连接到qml


    //写入到lru缓存 隐式同步


    //写入sql 跨进程同步

    //test
    cout<<"RT节点test "<<tmp.quest<<'\n';
}

template<>
void FixRun<NUMTHS, NUMRT, NUMVM>::LRUCQuery()
{
    using namespace std::this_thread;
    cout<<"Assist Threading lru query is using ...\n";
    //test
    return ;

    //单独窗口等待输入
    //根据退出信号
    while (!this->isdown.load()) {
    _cstr qust="";
    cout<<"cin keys to find lru keys\n";
    cout<<"cin aviliable key find lru answer\n";
    cout<<"cin: ";
    std::getline(std::cin, qust);

    //打印所有keys
    if (strcmp(qust.c_str(), "keys") == 0) {
        //内部线程隐式同步
        //可能造成计算线程写入等待
        this->lru->getkeys();
        continue;
    }
    
    _size key_tt=std::atoi(qust.data());
    if (key_tt < 0 || key_tt >= 999) { 
        cerr<<"\n查询的key在0~999 重新输入\n"; 
        continue;
    }
    //查找lru缓存 是否命中
    else {
    _usize key_utt=_usize(key_tt);
    //内部线程隐式同步 可能计算线程写入等待
    auto cache=this->lru->find(key_utt);

    if (cache == nullptr) {
    cerr<<"\n未命中lru 不存在该key对应的节点\n"; 
    continue;
    }
    else {
    cout<<"\n命中lru key "<<key_tt<<'\n';
    cout<<"question: "<<cache->quest<<'\n';
    cout<<"answer: "<<cache->answer<<'\n';
    cout<<"cycle continue\n";
    continue;
    }
    }

    //对于未知输入
    cerr<<"\nunkown cin and continue\n"; 
    continue;
    }
    //窗口关闭

    cout<<"FIXRun对象 LRUCQuery函数结束 !\n";
    return ;
}
