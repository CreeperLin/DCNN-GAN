#!/bin/sh

fname='fmri_data.zip'
dir=`dirname $0`
dname="$dir/fmri_data"
if [ -d $dname ]; then
    echo "$dname has already been downloaded."
    exit
fi
url='https://jbox.sjtu.edu.cn:10081/v2/delivery/data/fc060b36743c4d6d8b30fb03c130e59d/?token='
echo $url
if [ -f $fname ]; then
    echo "$fname has already been downloaded."
else
    curl -o "$dir/$fname" $url
fi
unzip "$dir/$fname" -d $dir
rm -rf "$dir/$fname"
