import os
import pandas as pd
import uuid
from dotenv import load_dotenv

load_dotenv()

DEFAULT_USERNAME = os.environ["DEFAULT_USERNAME"]
DEFAULT_PWD = os.environ["DEFAULT_PASSWORD"]
AUTH_FILE = os.environ.get("AUTH_FILE")

accounts = {
    DEFAULT_USERNAME: DEFAULT_PWD
}

try:
    account_df = pd.read_csv(AUTH_FILE)
    accounts.update({account_df["username"][i]: account_df["password"][i] for i in range(len(account_df))})
except Exception:
    pass


def default_auth(username, password):
    return username in accounts and accounts[username] == password
        