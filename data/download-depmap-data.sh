#!/bin/bash

# Download DepMap data.

## VARIABLES
# year and quarter of data release
YEAR_Q="20q3"
# FigShare URL
URL="https://ndownloader.figshare.com/articles/12931238/versions/1"


## Derived values (need not change).
DEPMAP_NAME="depmap_${YEAR_Q}"
ZIP_NAME="${DEPMAP_NAME}.zip"


## Download data
mkdir $DEPMAP_NAME
cd $DEPMAP_NAME || exit
curl -o $ZIP_NAME $URL
unzip $ZIP_NAME
