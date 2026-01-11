from flask import Flask, request, jsonify
import sys
from collections import Counter, OrderedDict, defaultdict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
from time import time
from pathlib import Path
import pickle
import math
import numpy as np
from google.cloud import storage
import sqlite3
import subprocess

# ייבוא המחלקה מהקובץ הנוסף שחייב להיות באותה תיקייה
from inverted_index_gcp import InvertedIndex, MultiFileReader


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

BUCKET_NAME = '208129742'

# השמות של התיקיות בתוך הבאקט
REMOTE_BODY_INDEX_DIR = 'postings_gcp'  # התיקייה בבאקט שבה נמצא ה-index.pkl וה-bins
REMOTE_PR_FILE = 'pr/pr_index.pkl'  # הקובץ של ה-PageRank בתוך הבאקט
REMOTE_ID2TITLE_FILE = 'id2title.sqlite'  # קובץ המיפוי

# תיקיות מקומיות (ב-Colab/GCP)
LOCAL_BODY_INDEX_DIR = 'body_index_local'
LOCAL_ID2TITLE_PATH = 'id2title.sqlite'


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def setup_swap_memory():
    """
    מגדיר זיכרון וירטואלי (Swap) של 4GB כדי למנוע קריסות RAM.
    """
    print("Setting up Swap Memory...", flush=True)
    try:
        # בדיקה אם כבר קיים
        if os.path.exists('/swapfile'):
            print("Swapfile already exists. Skipping creation.", flush=True)
            return

        subprocess.run("sudo fallocate -l 4G /swapfile", shell=True, check=True)
        subprocess.run("sudo chmod 600 /swapfile", shell=True, check=True)
        subprocess.run("sudo mkswap /swapfile", shell=True, check=True)
        subprocess.run("sudo swapon /swapfile", shell=True, check=True)
        print("Swap memory configured successfully (4GB).", flush=True)
    except Exception as e:
        print(f"Warning: Failed to setup swap memory: {e}", flush=True)


def download_index_from_bucket(bucket_name, source_dir, dest_dir):
    """
    מוריד את כל קבצי האינדקס (pkl + bin) מהבאקט לדיסק המקומי.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    print(f"Downloading index from {bucket_name}/{source_dir} to {dest_dir}...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_dir)

    count = 0
    for blob in blobs:
        if blob.name.endswith(".pkl") or blob.name.endswith(".bin"):
            # חילוץ שם הקובץ בלבד (בלי התיקייה בבאקט)
            file_name = os.path.basename(blob.name)
            local_path = os.path.join(dest_dir, file_name)

            # מורידים רק אם לא קיים כבר
            if not os.path.exists(local_path):
                blob.download_to_filename(local_path)
            count += 1
            if count % 10 == 0:
                print(f"Downloaded {count} files...")
    print(f"Finished downloading {count} new files.")


def download_file_from_bucket(bucket_name, source_file, dest_file):
    if os.path.exists(dest_file):
        print(f"File {dest_file} already exists, skipping download.")
        return

    print(f"Downloading {source_file} from bucket...", flush=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_file)
    blob.download_to_filename(dest_file)
    print("Download complete.")


def load_pagerank(bucket_name, remote_path):
    """
    טוען את מילון ה-PageRank ישירות מהבאקט (ללא שמירה לדיסק).
    """
    print("Loading PageRank from Bucket...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(remote_path)
    if blob.exists():
        content = blob.download_as_bytes()
        return pickle.loads(content)
    else:
        print(f"Warning: PageRank file not found at {remote_path}")
        return {}


def get_titles_batch(doc_ids):
    """
    מקבל רשימה של מזהי מסמכים ומחזיר מילון של {id: title} מתוך ה-SQLite.
    """
    if not doc_ids:
        return {}

    titles_map = {}
    try:
        # חיבור למסד הנתונים
        conn = sqlite3.connect(LOCAL_ID2TITLE_PATH)
        cursor = conn.cursor()

        # בניית השאילתה עם Placeholders
        placeholders = ','.join(['?'] * len(doc_ids))

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        if not tables:
            print("Error: No tables found in sqlite db")
            return {}

        table_name = tables[0][0]

        # שאילתה לשליפת הכותרות (ממירים למחרוזת אם צריך)
        query = f"SELECT id, title FROM {table_name} WHERE id IN ({placeholders})"
        cursor.execute(query, tuple(doc_ids))

        results = cursor.fetchall()
        for doc_id, title in results:
            titles_map[int(doc_id)] = title

        conn.close()
    except Exception as e:
        print(f"Error fetching titles from SQLite: {e}")

    return titles_map


# טוקניזציה (זהה למטלה 1)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stopwords_frozen = frozenset(
    ['during', 'as', 'whom', 'no', 'so', 'shouldn\'t', 'she\'s', 'were', 'needn', 'then', 'on', 'should\'ve', 'once',
     'very', 'any', 'they\'ve', 'it\'s', 'it', 'be', 'why', 'ma', 'over', 'you\'ll', 'they', 'you\'ve', 'am', 'before',
     'shan', 'nor', 'she\'d', 'because', 'been', 'doesn\'t', 'than', 'will', 'they\'d', 'not', 'those', 'had', 'this',
     'through', 'again', 'ours', 'having', 'himself', 'into', 'i\'m', 'did', 'hadn', 'haven', 'should', 'above',
     'we\'ve', 'does', 'now', 'm', 'down', 'he\'d', 'herself', 't', 'their', 'hasn\'t', 'few', 'and', 'mightn\'t',
     'some', 'do', 'the', 'we\'re', 'myself', 'i\'d', 'won', 'after', 'needn\'t', 'wasn\'t', 'them', 'don', 'further',
     'we\'ll', 'hasn', 'haven\'t', 'out', 'where', 'mustn\'t', 'won\'t', 'at', 'against', 'shan\'t', 'has', 'all', 's',
     'being', 'he\'ll', 'he', 'its', 'that', 'more', 'by', 'who', 'i\'ve', 'o', 'that\'ll', 'there', 'too', 'they\'ll',
     'own', 'aren\'t', 'other', 'an', 'here', 'between', 'hadn\'t', 'isn\'t', 'below', 'yourselves', 've', 'isn',
     'wouldn', 'd', 'we', 'couldn', 'ain', 'his', 'wouldn\'t', 'was', 'didn', 'what', 'when', 'i', 'i\'ll', 'with',
     'her', 'same', 'you\'re', 'yours', 'couldn\'t', 'for', 'doing', 'each', 'aren', 'which', 'such', 'mightn', 'up',
     'mustn', 'you', 'only', 'most', 'of', 'me', 'she', 'he\'s', 'in', 'a', 'if', 'but', 'these', 'him', 'hers', 'both',
     'my', 'she\'ll', 're', 'weren', 'yourself', 'is', 'until', 'weren\'t', 'to', 'are', 'itself', 'you\'d',
     'themselves', 'ourselves', 'just', 'wasn', 'have', 'don\'t', 'll', 'how', 'they\'re', 'about', 'shouldn', 'can',
     'our', 'we\'d', 'from', 'it\'d', 'under', 'while', 'off', 'y', 'doesn', 'theirs', 'didn\'t', 'or', 'your',
     'it\'ll'])


def tokenize(text):
    return [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in stopwords_frozen]


class BM25_from_index:
    def __init__(self, index, k1=1.2, b=0.6):
        self.b = b
        self.k1 = k1
        self.index = index

        if hasattr(index, 'DL') and len(index.DL) > 0:
            self.N = len(index.DL)
            self.AVGDL = sum(index.DL.values()) / self.N
        else:
            self.N = 6348910
            self.AVGDL = 300

    def calc_idf(self, list_of_tokens):
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df:
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def search(self, query, N=100):
        self.idf = self.calc_idf(query)
        candidates_scores = defaultdict(float)

        file_requests = defaultdict(list)
        for term in query:
            if term in self.index.df:
                locs = self.index.posting_locs[term]
                for filename, offset in locs:
                    file_requests[filename].append((term, offset))

        for filename, requests in file_requests.items():
            full_path = os.path.join(LOCAL_BODY_INDEX_DIR, filename)
            try:
                with open(full_path, 'rb') as f:
                    for term, offset in requests:
                        f.seek(offset)
                        n_bytes = self.index.df[term] * 6
                        b = f.read(n_bytes)
                        for i in range(self.index.df[term]):
                            doc_id = int.from_bytes(b[i * 6:i * 6 + 4], 'big')
                            tf = int.from_bytes(b[i * 6 + 4:(i + 1) * 6], 'big')
                            candidates_scores[doc_id] += self._score(term, doc_id, tf)
            except FileNotFoundError:
                continue

        # מחזירים את כל המועמדים כדי שנוכל לנרמל אותם בחוץ
        return sorted(candidates_scores.items(), key=lambda item: item[1], reverse=True)

    def _score(self, term, doc_id, freq):
        if hasattr(self.index, 'DL'):
            doc_len = self.index.DL.get(doc_id, self.AVGDL)
        else:
            doc_len = self.AVGDL
        numerator = self.idf[term] * freq * (self.k1 + 1)
        denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
        return numerator / denominator


# 1. הפעלת מנגנון Swap (Virtual Memory) לפני הכל
setup_swap_memory()

idx_body = None
pr_dict = {}

# הטעינה מתבצעת רק כשהאפליקציה עולה
print("Starting up application...", flush=True)

# 2. הורדת הקבצים מהבאקט
try:
    # הורדת האינדקס
    if not os.path.exists(LOCAL_BODY_INDEX_DIR) or not os.listdir(LOCAL_BODY_INDEX_DIR):
        print("Downloading index files...", flush=True)
        download_index_from_bucket(BUCKET_NAME, REMOTE_BODY_INDEX_DIR, LOCAL_BODY_INDEX_DIR)
    else:
        print("Index files found locally, skipping download.", flush=True)

    # הורדת ה-SQLite (ID to Title)
    download_file_from_bucket(BUCKET_NAME, REMOTE_ID2TITLE_FILE, LOCAL_ID2TITLE_PATH)

except Exception as e:
    print(f"CRITICAL ERROR downloading files: {e}", flush=True)

# 3. טעינת האינדקס לזיכרון
print("Loading Body Index to memory...", flush=True)
try:
    idx_body = InvertedIndex.read_index(base_dir=LOCAL_BODY_INDEX_DIR, name='index')

    # --- בדיקות שפיות קריטיות ---
    print(f"SUCCESS: Body Index Loaded.", flush=True)

except Exception as e:
    print(f"CRITICAL ERROR loading Body Index: {e}", flush=True)
    idx_body = None

# 4. טעינת PageRank
try:
    pr_dict = load_pagerank(BUCKET_NAME, REMOTE_PR_FILE)
    print("PageRank Loaded.")
except Exception as e:
    print(f"Error loading PageRank: {e}")

# =========================================================================
# ROUTES
# =========================================================================
# חישוב המקסימום של ה-PageRank פעם אחת (גלובלי) כדי לנרמל
max_pr = 1.0
if pr_dict:
    max_pr = max(pr_dict.values())


@app.route("/search")
def search():
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    query_tokens = tokenize(query)

    if idx_body:
        bm25 = BM25_from_index(idx_body)

        # 1. קבלת כל המועמדים (ללא הגבלת N בשלב ראשון כדי למצוא מקסימום אמיתי)
        candidates = bm25.search(query_tokens, N=1000)

        if not candidates:
            return jsonify([])

        max_bm25_score = candidates[0][1]  # הרשימה ממוינת, הראשון הוא הגבוה ביותר
        if max_bm25_score == 0: max_bm25_score = 1

        final_results = []

        if len(query_tokens) <= 2:
            w_bm25 = 0.3
            w_pagerank = 0.7
        else:
            w_bm25 = 0.9
            w_pagerank = 0.1

        # לוקחים רק את ה-100 הכי טובים מה-BM25
        top_candidates = candidates[:100]

        for doc_id, score in top_candidates:
            # נירמול BM25 לטווח 0-1
            norm_bm25 = score / max_bm25_score

            # נירמול PageRank לטווח 0-1
            pr_val = pr_dict.get(doc_id, 0)
            norm_pr = pr_val / max_pr

            # שילוב חכם
            new_score = (w_bm25 * norm_bm25) + (w_pagerank * norm_pr)
            final_results.append((doc_id, new_score))

        final_results.sort(key=lambda x: x[1], reverse=True)

        # 1. איסוף כל ה-IDs שצריך להם כותרת
        doc_ids_to_fetch = [doc_id for doc_id, score in final_results]

        # 2. שליפת הכותרות מהדאטה-בייס המקומי
        titles_map = get_titles_batch(doc_ids_to_fetch)

        # 3. יצירת הרשימה הסופית (id, title)
        # אם אין כותרת, נחזיר את ה-ID כמחרוזת כברירת מחדל
        res = [(str(doc_id), titles_map.get(doc_id, str(doc_id))) for doc_id, score in final_results]

        return jsonify(res)

    return jsonify([])


@app.route("/search_body")
def search_body():
    return jsonify([])


@app.route("/search_title")
def search_title():
    return jsonify([])


@app.route("/search_anchor")
def search_anchor():
    return jsonify([])


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    res = [pr_dict.get(doc_id, 0) for doc_id in wiki_ids]
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    res = [0 for _ in wiki_ids]
    return jsonify(res)


def run(**options):
    app.run(**options)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)