#include "process.h"

using std::cout;
using std::cerr;
using std::endl;
using namespace Alias::Sems;
namespace py = pybind11;
/* Run 定义进程函数 */
template<>
void Run<NOR_PROCS>::QMLgui(_cstr sem_name)
{
    //获取已创建的信号量
    auto rt_sem=sem_open(sem_name.data(), SEM_GET);

    cout<<endl<<"QMLgui 进程进入 pid "<<getpid()<<endl;
    
    //启用qt支持
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    int argc=0;
    char *argv[]={};
    QGuiApplication app(argc, argv);
    
    QSLwirte slWriter;
    QLUwrite luWriter;
    
    //注册到qml
    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty("slWriter", &slWriter);
    engine.rootContext()->setContextProperty("luWriter", &luWriter);
    
    // 加载QML文件
    auto url_str=QString::fromStdString(qml_file);
    const QUrl url(url_str);
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
                     &app, [url](QObject *obj, const QUrl &objUrl) {
        if (!obj && url == objUrl)
            QCoreApplication::exit(-1);
    }, Qt::QueuedConnection);
    engine.load(url);

    app.exec();
    return ;
} 

template<>
void Run<NOR_PROCS>::RTrunning(_cstr sem_name)
{
    //获取已创建的信号量
    auto rt_sem=sem_open(sem_name.data(), SEM_GET);

    cout<<endl<<"RT 推理进程省略 ! pid "<<getpid()<<endl;

    //等待结束信号 阻塞等待
    if (sem_wait(rt_sem) == -1) {
    cerr<<"rt子进程等待信号量错误 强制停止\n";
    exit(1);
    }

    cout<<"RT 推理进程结束 !\n";

    exit(EXIT_SUCCESS);
}

template<>
void Run<NOR_PROCS>::VMrunning(_cstr sem_name)
{
    //获取已创建的信号量
    auto vm_sem=sem_open(sem_name.data(), SEM_GET);

    cout<<endl<<"TVM 推理进程省略 ! pid "<<getpid()<<endl;

    //等待结束信号 阻塞等待
    if (sem_wait(vm_sem) == -1) {
    cerr<<"vm子进程等待信号量错误 强制停止\n";
    exit(1);
    }

    cout<<"TVM 推理进程结束 !\n";

    exit(EXIT_SUCCESS);
}

template<>
void Run<NOR_PROCS>::FXrunning(_cstr sem_name)
{
    //获取已创建的信号量
    auto vm_sem=sem_open(sem_name.data(), SEM_GET);
    //子进程中单独初始化Python解释器
    //py::scoped_interpreter guard{}; 
    //py::module_ sys=py::module_::import("sys");
    //sys.attr("path").attr("append")("..");

    cout<<endl<<"FIX 推理进程进入 ! pid "<<getpid()<<endl;

    //构造中执行
    this->fixrun_=new FixRun<NUMTHS, NUMRT, NUMVM>();

    //等待结束信号 阻塞等待
    if (sem_wait(vm_sem) == -1) {
    cerr<<"vm子进程等待信号量错误 强制停止\n";
    exit(1);
    }
    
    //异步通知停止
    this->fixrun_->rundow();
    //子进程析构推理对象
    delete this->fixrun_;
    sleep(1);
    cout<<"FIX 推理进程结束 !\n";

    exit(EXIT_SUCCESS);
}

template<>
void Run<NOR_PROCS>::SQfinding(_cstr sem_name)
{
    //获取已创建的信号量
    _sem sq_sem=sem_open(sem_name.data(), SEM_GET);

    cout<<endl<<"SQL 查询进程进入 pid "<<getpid()<<endl;
    //阻塞等待
    sq_op->quest_find(sq_op);

    //等待结束信号 阻塞等待
    if (sem_wait(sq_sem) == -1) {
    cerr<<"sql查找子进程等待信号量错误 强制停止\n";
    exit(1);
    }

    cout<<"SQL 查询进程结束 !\n";
    sleep(4);
    cout<<"任务退出之前写入csv文件 !\n";

    //结束后sql逐条加载
    
    
    //所有row使用脚本写入csv
    

    exit(EXIT_SUCCESS);
}
