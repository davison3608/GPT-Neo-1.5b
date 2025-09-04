#include "sqconnect.h"
#include "qtfilewirte.h"
#include <future> 
#include <fstream>
#include <unistd.h>

using namespace Alias;
using std::string;
using std::cout;
using std::cerr;
using std::ofstream;
using std::ifstream;

/* termial对象 */
terminal::terminal(string tri_name)
{
    //终端名 
    this->tmp_ter=tri_name;
    //创建临时文件
    this->tmp_file=this->tmp_ter + "_tmp.txt";
    std::ofstream{this->tmp_file};
}

terminal::~terminal()
{
    std::ifstream file(this->tmp_file);
    if (file.is_open())
    std::remove(this->tmp_file.c_str());
    //删除终端输出文件
    file.close();
}

void terminal::terminalshow()
{
    Alias::_cstr cmd="gnome-terminal --title=" + tmp_ter + " -- bash -c 'while true; do "
        "if [ -s " + tmp_file + " ]; then "  // 若文件有内容
        "cat " + tmp_file + "; "             // 打印内容
        "echo > " + tmp_file + "; "          // 清空文件
        "fi; "
        "sleep 0.5; "
        "done' &";
    system(cmd.c_str());
    usleep(500000); //等待启动完成
}

void terminal::terminalout(string& str_ou)
{
    // 写入临时文件 追加模式
    std::ofstream ofs(tmp_file, std::ios::app);
    if (ofs) {
    ofs << str_ou << std::endl;
    }
}

const string lru_tmp="../tmp/tmp_lru.txt";
const string sql_tmp="../tmp/tmp_sql.txt";
const string qml_file="../findwidget.qml";

bool terminal::terminalcome(string file, string cmd)
{
    //打开文件 
    ifstream infile(file);
    if (!infile.is_open()) {
    cerr<<"终端交互文件不存在 "<<file<<'\n';
    return false;
    }
    //读取 一旦有数据则匹配后返回
    bool status;
    while (1) {
    //空则重复
    if (infile.peek() == ifstream::traits_type::eof()) {
    std::this_thread::yield();
    continue;
    }
    //存在则读取 
    else {
    infile.seekg(0, std::ios::end);
    auto length=infile.tellg();
    infile.seekg(0, std::ios::beg);
    char in_str[length];
    infile.read(in_str, length);
        
    if (strcmp(in_str, cmd.c_str()) == 0) {
    //清空后返回
    ofstream oufile(file);
    oufile.close();
    infile.close();

    return true;
    }
    else {
    //清空后返回
    ofstream oufile(file);
    oufile.close();
    infile.close();
    return false;
    }
    }
    }
}
