#include "qtfilewirte.h"
#include <future> 
#include <fstream>
#include <unistd.h>

using std::string;
using std::cout;
using std::cerr;
using std::ofstream;
using std::ifstream;

void QSLwirte::write_to(QString str)
{
    auto c_str=str.toStdString();
    ofstream file(sql_tmp);
    if (file.is_open()) {
    file << c_str;
    return ;
    }
    else {
    cerr<<"无法打开文件 "<<sql_tmp<<'\n';
    return ;
    }
}

void QLUwrite::write_to(QString str)
{
    auto c_str=str.toStdString();
    ofstream oufile(lru_tmp);
    if (oufile.is_open()) {
    oufile << c_str;
    return ;
    }
    else {
    cerr<<"无法打开文件 "<<lru_tmp<<'\n';
    return ;
    }
}

