# 当传入有参数时，删除参数所指定的文件
# 当传入无参数时，删除所有文件

if [ $# -eq 0 ]
then
    rm *.txt
    rm *.pkl
else
    for arg in "$@";do
        if [ ! -d "$arg" ];
        then
            echo "$arg not exist"
        else
            rm $arg
            echo "$arg cleaned up"
        fi
    done
fi