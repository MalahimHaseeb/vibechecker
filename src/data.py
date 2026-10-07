import time
import urllib.request
from pathlib import Path

import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "reddit.csv"


def load_data(retries=5):
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(exist_ok=True)
        for attempt in range(retries):
            try:
                urllib.request.urlretrieve(DATA_URL, DATA_PATH)
                break
            except Exception:
                DATA_PATH.unlink(missing_ok=True)
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError("could not download the dataset after several tries")
    return pd.read_csv(DATA_PATH)