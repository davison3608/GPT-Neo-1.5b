#include "strpackets.h"
#include "sqconnect.h"
#include "runpackets.h"
#include "process.h"

using std::cerr;
using std::cout;
using namespace Alias;
using namespace Alias::Sems;
using namespace Alias::RT;
namespace py = pybind11;

using namespace std::this_thread;
using namespace std::chrono;

/* main入口 */
extern "C"
int main(int argc, char *argv[])
{
cout<<"父进程执行 pid "<<getpid()<<'\n';

////启用qt支持
//QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
//QGuiApplication app(argc, argv);
//
//QSLwirte slWriter;
//QLUwrite luWriter;
//
////注册到qml
//QQmlApplicationEngine engine;
//engine.rootContext()->setContextProperty("slWriter", &slWriter);
//engine.rootContext()->setContextProperty("luWriter", &luWriter);
//
//// 加载QML文件
//auto url_str=QString::fromStdString(qml_file);
//const QUrl url(url_str);
//QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
//                 &app, [url](QObject *obj, const QUrl &objUrl) {
//    if (!obj && url == objUrl)
//        QCoreApplication::exit(-1);
//}, Qt::QueuedConnection);
//engine.load(url);

//app.exec();



//初始化全局python解释器
py::scoped_interpreter guard{};
//为解释器添加上一级路径
py::module_ sys = py::module_::import("sys");
sys.attr("path").attr("append")("..");

//main提前静态操作
SqliteCR::init_connect();
sleep_for(milliseconds(500));

//父进程持有Cess对象
Cess<NOR_PROCS> *Mcess=new Cess<NOR_PROCS>;   
sleep_for(milliseconds(500));    

//父进程获取所有函数包装对象
auto qml_pro=Mcess->getfunc(&Run<NOR_PROCS>::QMLgui);
auto sq_pro=Mcess->getfunc(&Run<NOR_PROCS>::SQfinding);
auto fix_pro=Mcess->getfunc(&Run<NOR_PROCS>::FXrunning);
auto rt_pro=Mcess->getfunc(&Run<NOR_PROCS>::RTrunning);
auto vm_pro=Mcess->getfunc(&Run<NOR_PROCS>::VMrunning);
    
//顺序并发 同时权限转移到形参
Mcess->funcfork(std::move(sq_pro));
sleep_for(milliseconds(500));

//单一推理进程
Mcess->funcfork(std::move(rt_pro));
Mcess->funcfork(std::move(vm_pro));
//混合推理
sleep_for(milliseconds(500));
Mcess->funcfork(std::move(fix_pro));

//父进程阻塞 等待所有子进程
//指令退出
sleep_for(seconds(20));
//gui
sleep_for(milliseconds(500));
Mcess->funcfork(std::move(qml_pro));
//等待问题
Mcess->exec();
//等待结束
Mcess->blockwait();

//所有子进程结束后析构
return 0;
}