#!/bin/sh

DIR=""

if [ $# -eq 0 ]
then
	DIR="**"
else
	DIR=$1
fi

for ipynb in $DIR/*.ipynb
do
    jupyter nbconvert --to markdown $ipynb
done
