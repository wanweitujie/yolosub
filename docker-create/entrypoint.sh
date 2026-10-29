#!/bin/bash

START_SCRIPT="/docker-data-yolo/start.sh"
STOP_SCRIPT="/docker-data-yolo/stop.sh"
if [ ! -f "${START_SCRIPT}" ]; then
    echo "Error: start.sh not found"
    exit 1
fi
chmod +x ${START_SCRIPT}
chmod +x ${STOP_SCRIPT}
bash ${START_SCRIPT}

exec "$@"