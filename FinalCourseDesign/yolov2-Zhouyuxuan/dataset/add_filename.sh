#!/bin/bash

cd ./images
for file in `ls .`
do
	echo $file >> filename.txt
done
