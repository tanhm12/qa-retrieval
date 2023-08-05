<p align="center">Question Answering over Documents with Langchain, OpenAI and Qdrant  </p>  

Features:  
1. Simple workflow: uploading documents --> processing documents --> QA session.   
2. Multiple users with basic authentication.  
3. Stable database (with Qdrant), support persisting and restoring processed documents.  
4. Intuitive UI, easy to add new components (thanks to Gradio).

# Running locally
## Prerequisites
1. Python 3.10 or higher (3.7 or above seem to be fine, but not being tested yet)
## Installation
### 1. Install python packages  
```bash
pip install -r requirements.txt
```
### 2. Add a new config file  
Copy the content of `.env.example` into `.env` file and put your own OpenAI key in it.
### 3. Multiple users (Optional)
For using with multiple users, you might want to add new users into the auth file (default is `auth.csv`).
## Running
### 1. Run Qdrant
You can run Qdrant locally or using docker. Below commands is used to run Qdrant with docker:  
```bash
cd docker
sh run_qdrant.sh
```
Qdrant is noww running at port 6333, if you change the port then you need to change the variable `QDRANT_URL` in the config file (`.env`).
### 2. Run service
Default running port is 10011 (can be changed in `demo.py`).
```bash
python demo.py
```

# Running with docker (recommended)
## Prerequisites
1. Docker version 20.10.22 or above
## Installation
Just need to build docker image.  
```bash
sh docker/build.sh
```
## Running
### 1. Configuration
- Change working directory to   `docker`.
- Copy the content of `docker-compose.yml.example` into `docker-compose.yml`.  
- Add your OpenAI key to this file, change other configs if needed.
- For multiple users: mount your auth.csv file (` "./auth.csv: /app/auth.csv"`)
### 2. Run
```bash
sh run.sh
```
Demo is now running at http://localhost:10011.

# Sample run (TODO: video)