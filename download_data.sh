#!/bin/sh

# Author: Shaun Kim
# Downloads ocean pCO2 testbed datasets 

dir=$(pwd)
TARGET_DIR="tmp.tar.gz"
BASE_URL="https://ndownloader.figshare.com/files"


cd ${dir}/data

echo "file downloading..."
wget ${BASE_URL}/${1:-"16130027"} -O ${TARGET_DIR}


echo "file unzipping..."
tar -xf ${TARGET_DIR}  

rm ${TARGET_DIR}
echo "files downloaded"
