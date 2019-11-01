#!/bin/sh
docker run -it --rm -v "$(pwd)":/home/script -v "$(pwd)/../data":/home/data efimov/wildfire python /home/script/train.py