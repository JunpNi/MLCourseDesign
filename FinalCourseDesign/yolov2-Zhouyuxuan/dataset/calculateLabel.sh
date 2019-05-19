#!/bin/bash

echo $1 `awk -v a=$2 -v b=$4 'BEGIN{printf "%.6f",a-b/2}'` `awk -v a=$3 -v b=$5 'BEGIN{printf "%.6f",a-b/2}'` `awk -v a=$2 -v b=$4 'BEGIN{printf "%.6f",a+b/2}'`  `awk -v a=$3 -v b=$5 'BEGIN{printf "%.6f",a+b/2}'` >> ../collectedLabelxxyy.txt
