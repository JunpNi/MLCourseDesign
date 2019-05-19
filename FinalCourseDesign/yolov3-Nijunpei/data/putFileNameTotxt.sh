function getdir(){
    echo $1
    for file in $1/*
    do
    if test -f $file
    then
        echo "data/$file" >> /Users/diangroup/Desktop/yolov3-master/data/tiny_vid_test.txt
        arr=(${arr[*]} $file)
    else
        getdir $file
    fi
    done
}
getdir coco/images
