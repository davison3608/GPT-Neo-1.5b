#include "runpackets.h"

using namespace Alias;
using namespace Alias::RT;
using std::cout;
using std::cerr;
using namespace std::this_thread;
using namespace std::chrono;

/* rtnode节点 */
rtNode::rtNode() noexcept:
context(nullptr), buffer({})
{
    //初始化

}

void rtNode::init(/* _Nvengine *engine */)
{
//
//
//
//
//
    //test
    sleep_for(milliseconds(500));
    cout<<"推理节点 RT Node is init ...\n";
}

void rtNode::destroy()
{
//
//
//
//  
    //test  
    sleep_for(milliseconds(500));
    cout<<"推理节点 RT Node is destroy !\n";
}

/* vmnode节点 */
vmNode::vmNode() noexcept:
buffer({})
{
    //初始化

} 

void vmNode::init()
{
//
//
//
//    
    //test
    sleep_for(milliseconds(500));
    cout<<"推理节点 TVM Node is init ...\n";
}

void vmNode::destroy()
{
//
//
//
    //test    
    sleep_for(milliseconds(500));
    cout<<"推理节点 TVM Node is destroy !\n";
}
