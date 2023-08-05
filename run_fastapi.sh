SERVE_PATH=/demo/qa gunicorn demo:app --workers 1 --worker-class uvicorn.workers.UvicornWorker \
     --bind 0.0.0.0:10011 --log-level debug --error-logfile log/server.log

# SERVE_PATH=/demo/qa uvicorn demo:app --workers 1 --host 0.0.0.0 --port 10011 --log-level debug 