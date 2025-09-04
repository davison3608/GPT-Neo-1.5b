#ifndef DOUBRUN
#define DOUBRUN
//! rt推理节点
#define NUMRT     2
//! vm推理节点
#define NUMVM     2
//! 计算线程数目
#define NUMTHS    6

//! 基础结构
#include "strpackets.h"
//! 文件加载对象
#include "sopackets.h"
//! sql接口
#include "sqconnect.h"

namespace Alias {
    using precision_ou = _Float32;
    using _ste = std::chrono::time_point<std::chrono::system_clock>;
    using _buf = void*;

    using _1d_shapes = std::array<precison, 1>;
    using _2d_shapes = std::array<precison, 2>;
    using _1df_shapes = std::array<precision_ou, 1>;
    using _2df_shapes = std::array<precision_ou, 2>;
    using _thread = std::thread;
    using _vec_teads = std::vector<std::thread>;

namespace RT {
    using _Nvengine = nvinfer1::ICudaEngine;
    using _NVrun = nvinfer1::IRuntime;
    using _Nvcont = nvinfer1::IExecutionContext;    
    using _buf = void*;
} // namespace RT

namespace IO {
    using _in_tuple = std::tuple<Alias::_cstr, Alias::_1d_shapes>;
    using _ou_tuple = std::tuple<Alias::_cstr, Alias::_2df_shapes>;
} // namespace IO
    
} // namespace Alias

//! py脚本接口
#include "embed.h"

#ifndef TRTRUN_
#define TRTRUN_
//! trt推理节点
struct rtNode 
{
    rtNode() noexcept;
    ~rtNode() noexcept = default;
    //! 是否空闲  
    Alias::_flag isfree;
    //! 推理上下文
    Alias::RT::_Nvcont *context;
    //! 节点信息
    Alias::_usize batch;
    Alias::IO::_in_tuple in_ids;
    Alias::IO::_in_tuple in_mask;
    Alias::IO::_ou_tuple ou_lits;
    //! 推理缓冲区
    Alias::RT::_buf buffer[3];
    Alias::RT::_cus stream;
    Alias::RT::_cue start;
    Alias::RT::_cue end;
    //! 手动初始化
    void init(/* Alias::RT::_Nvengine *engine */);
    //! 手动销毁
    void destroy();
};

//! trt计算对象
//! 省略
#endif //TRTRUN

#ifndef TVMRUN_
#define TVMRUN_
//! tvm推理节点
struct vmNode
{
    vmNode() noexcept;
    ~vmNode() noexcept = default;
    //! 是否空闲  
    Alias::_flag isfree;
    //! 推理上下文

    //! 节点信息
    Alias::_usize batch;
    Alias::IO::_in_tuple in_ids;
    Alias::IO::_in_tuple in_mask;
    Alias::IO::_ou_tuple ou_lits;
    //! 推理缓冲区
    Alias::_buf buffer[3];
    Alias::_ste start;
    Alias::_ste end;
    //! 手动初始化
    void init();
    //! 手动销毁
    void destroy();
};

//! tvm计算对象
//! 省略
#endif //TVMRUN

#ifndef FIXRUN_
#define FIXRUN_
//! 混合节点计算对象
//! 持有sql接口
template<uint RTS, uint VMS>
class baseFIXRun: public RTload, public Soload
{
protected:
    //! 共享内存对象 静态返回
    PrcsCache *share;
    //! 推理队列对象 计算线程锁
    std::unique_ptr<Que> que;
    //! 无需锁 隐式同步

    //! 计算线程锁 计算线程被通知信号
    Alias::_mtx mt_run;
    Alias::_cv cv_run;

    //! 推理节点组
    rtNode rt_no[RTS];
    vmNode vm_no[VMS];
    //! 存在空闲节点标志 隐式同步改变
    Alias::_flag isnode;

protected:
    //! lru对象
    std::unique_ptr<LRUC<NOR_HASHNODE, NOR_CAPACITY>> lru;
    //! 无需锁 隐式同步

protected:
    //! sql接口 静态返回
    SqliteCR *sq_op;
    //! 无需访问锁 接口隐式同步

public:
    /**
     * 构造队列 构造lru
     * 手动构造所有推理节点
     * 静态获取sql接口
     * 手动构造所有推理节点
     * 启用que写入线程 que队首监控线程 节点监控线程 
    */
    baseFIXRun();
    /**
     * sql对象引用递减
     * 推理节点手动销毁
     * 释放队列对象 lru对象
    */
    virtual ~baseFIXRun();

    virtual void ContxSar() {}

    virtual void RTInference() {}

    virtual void VMInference() {}

private:
    //! que写入
    void Shareanydata();
    Alias::_thread t_share;
    //! que队首监控
    void Queanydata();
    Alias::_thread t_que;
    //! node空闲监控
    void Nodeanyfree();
    Alias::_thread t_node;

protected:
    //! 线程池退出标志
    //! 原子变量 隐式同步
    Alias::_flag isdown;
};

//! 实际推理对象
//! 并发所有推理节点
template<uint NUMDS, uint RTS, uint VMS>
class FixRun: public baseFIXRun<RTS, VMS>
{
protected:
    //! 线程序列
    Alias::_vec_teads threads;

    //! 解码器接口
    std::unique_ptr<deser> pydeser;
    
public:
    /**
     * 构造解码器对象 启用缓存查询函数
     * 并发计算线程后返回 
     * 需要外部阻塞
    */
    FixRun();
    ~FixRun();

    //! 外部调用 所有线程退出
    void rundow();

    using baseFIXRun<RTS, VMS>::RTInference;
    using baseFIXRun<RTS, VMS>::VMInference;

private:
    //! 线程实际执行函数
    void ContxSar() override;

private:
    //! rt节点计算函数
    void RTInference(rtNode *&node, queNode& tmp);

    //! vm节点计算函数
    void VMInference(vmNode *&node, queNode& tmp);

    //! lru查询函数
    void LRUCQuery();
};

#endif //FIXRUN

#endif //DOUBRUN