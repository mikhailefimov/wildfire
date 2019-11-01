#!/bin/sh
JUPYTER_PORT=8888
TOKEN=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 48)
(sleep 10 && xdg-open http://127.0.0.1:${JUPYTER_PORT}?token=${TOKEN}) &
docker run --rm -it -p ${JUPYTER_PORT}:8888 -v "$(pwd)":/home  -e DATASETS_PATH="/home/data" efimov/wildfire jupyter notebook  --notebook-dir=/home --no-browser --ip 0.0.0.0 --allow-root --NotebookApp.token="'${TOKEN}'"