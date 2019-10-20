#!/bin/sh
docker run -it --rm -v "$(pwd)":/home/script -v "$(pwd)/../data":/home/data -e DATASETS_PATH="/home/data" sbpython python /home/script/train.py