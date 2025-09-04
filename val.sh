# 假设可执行文件名为 talkterminal，当前目录为 build 目录
valgrind --leak-check=full --show-leak-kinds=all --log-file=./build/valbug.txt ./build/talkterminal
