import pandas as pd
import urllib.request
from concurrent.futures import ThreadPoolExecutor

def accessions(csv_path):
    return pd.read_csv(csv_path)["protein_id"].dropna().unique().tolist()

def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    with urllib.request.urlopen(req) as response:
        return response.read()

def pmap(func, iterable, max_workers):
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(func, iterable))
