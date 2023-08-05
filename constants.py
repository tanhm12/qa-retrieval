import os
from dotenv import load_dotenv

load_dotenv()
SERVE_PATH = os.environ.get('SERVE_PATH', "/")
QUEUE_SIZE = int(os.environ.get('QUEUE_SIZE', "3"))
