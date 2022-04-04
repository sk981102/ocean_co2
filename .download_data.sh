#!/bin/sh

# Author: Shaun Kim
# Downloads ocean pCO2 testbed datasets 

dir=$(pwd)
TARGET_DIR="data/tmp.tar.gz"
BASE_URL="https://ndownloader.figshare.com/files"


cd ${dir}

echo "file downloading...\n"
wget ${BASE_URL}/${1:-"16130027"} -O ${TARGET_DIR}

echo "file unzipping...\n"
tar -xf ${TARGET_DIR}


echo "files downloaded"
