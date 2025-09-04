#ifndef QT_WIRTE
#define QT_WIRTE
#include "iostream"
#include "string"

#include <QGuiApplication>
#include <QtQml/qqml.h>
#include <QtQml/QQmlApplicationEngine>
#include <QtQml/QQmlContext>
#include <Qt>
#include <QtCore/QObject>

//! 临时文件路径
extern const std::string lru_tmp;
extern const std::string sql_tmp;
extern const std::string qml_file;

//! qml调用对象 写入临时txt文件
class QSLwirte: public QObject
{
    Q_OBJECT
public:
    QSLwirte(QObject *parent=nullptr)
    {}
    ~QSLwirte() = default;

    Q_INVOKABLE void write_to(QString str);
};

class QLUwrite: public QObject
{
    Q_OBJECT
public:
    QLUwrite(QObject *parent=nullptr)
    {}
    ~QLUwrite() = default;

    Q_INVOKABLE void write_to(QString str);
};

//! qml文件加载
static void qmlexec(std::string qml_file);

#ifndef NEW_TERMIAL
#define NEW_TERMIAL
//! 终端执行对象 临时文件交互
class terminal
{
private:
    //! 临时文件用于显示
    std::string tmp_file;
    //! 临时终端名称
    std::string tmp_ter;

public:
    terminal(): terminal("new_termial")
    {}
    terminal(std::string tri_name);
    ~terminal();

    //! 新终端显示
    void terminalshow();

    //! 终端打印
    void terminalout(std::string& str_ou);

    //! 阻塞 
    bool terminalcome(std::string file, std::string cmd);
};

#endif

//! 取消宏
#undef slots

#endif // QT WIRTE