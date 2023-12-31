SOURCE=.
ENTRY=torch_main.py
CONFIG=config
DEST=./mlops
IGNORE=__pycache__,*.git,cache_dir,"${DEST}"
mkdir -p "${DEST}"
fedml build -t client -sf $SOURCE -ep $ENTRY -cf $CONFIG -df $DEST --ignore $IGNORE


SOURCE=.
ENTRY=torch_main.py
CONFIG=config
DEST=./mlops
IGNORE=__pycache__,*.git,cache_dir,"${DEST}"
mkdir -p "${DEST}"
fedml build -t server -sf $SOURCE -ep $ENTRY -cf $CONFIG -df $DEST --ignore $IGNORE
