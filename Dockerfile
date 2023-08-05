FROM python:3.10.6 as builder
WORKDIR /app
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"
COPY ./requirements.txt ./
RUN pip install --no-cache-dir gunicorn loguru
RUN pip install --no-cache-dir -r requirements.txt


FROM python:3.10.6-slim
WORKDIR /app
COPY --from=builder  /app/venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

COPY . ./
RUN cp .env.example .env
RUN mkdir log
### CMD gunicorn demo:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:10011 --log-level debug --error-logfile log/server.log
CMD python demo.py


# FROM python:3.10.6 as builder
# WORKDIR /app
# RUN python -m venv venv
# ENV PATH="/app/venv/bin:$PATH"
# COPY ./requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install gunicorn loguru
# COPY . ./
# RUN cp .env.example .env
# RUN mkdir log
# # CMD gunicorn demo:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:10011 --log-level debug --error-logfile log/server.log
# CMD python demo.py



