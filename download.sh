#!/bin/bash

set -e

DIR_SCRIPT="$( cd "$(dirname "$0")" ; pwd -P )"
DOWNLOAD_PATH="$DIR_SCRIPT/data"

## 2019 MeSH descriptors and supplemental concept records
echo "- Downloading MeSH descriptors and supplemental concept records"
DIR_MESH="$DOWNLOAD_PATH/mesh"
if [ ! -d "$DIR_MESH" ]; then
    mkdir -p "$DIR_MESH"
fi
wget -P "$DIR_MESH" -c "https://mir.jiho.us/bmet/desc2019.gz"
echo "    + Decompressing gzip files..."
cd $DIR_MESH
gzip -d "desc2019.gz"
cd "$DIR_SCRIPT"

# Download PubTator
echo "- Downloading PubTator data file"
DIR_PUB="$DOWNLOAD_PATH/pubtator"
if [ ! -d "$DIR_PUB" ]; then
    mkdir -p "$DIR_PUB"
fi
wget -P "$DIR_PUB" -c "https://mir.jiho.us/data/bmet/bioconcepts2pubtator_offsets.gz"
cd "$DIR_PUB"
echo "    + Decompressing gzip files..."
gzip -d "bioconcepts2pubtator_offsets.gz"
cd "$DIR_SCRIPT"
