#!/bin/bash
cd labels
for file in `ls`
do
	../calculateLabel.sh `cat $file`
done

