# Wikipedia Search Engine

This project implements a search engine for the English Wikipedia corpus, deployed on Google Cloud Platform (GCP).

## Project Overview
The search engine is a Flask-based web application. It retrieves relevant documents using a pre-computed inverted index stored in a Google Cloud Storage bucket. The ranking mechanism combines textual similarity (BM25) and static page importance (PageRank).

## File Structure

* `search_frontend.py`: The main entry point. Initializes the Flask app, handles memory configuration (swap), loads indices, and processes search requests.
* `inverted_index_gcp.py`: Utility class for reading the inverted index, including handling posting lists and binary file formats.
* `id2title.sqlite`: Local SQLite database mapping document IDs to titles (downloaded at runtime).

## Implementation Details

### Data Storage
All index data is hosted in a public Google Cloud Storage bucket (`gs://208129742`).
* **Inverted Index:** Stored as binary files in the `postings_gcp/` directory.
* **PageRank:** Stored in `pr/pr_index.pkl`.
* **Titles:** Managed via a lightweight SQLite database to reduce memory usage.

### System Configuration
To handle the large index within the memory constraints of a standard VM, the application automatically configures 4GB of swap memory upon startup.

### Retrieval & Ranking
The search engine utilizes a two-stage process:
1. **Candidate Retrieval:** Documents are retrieved using the body index.
2. **Scoring:** The final score is a linear combination of BM25 and PageRank. The weights are dynamic based on query length:
   * **Short queries (1-2 words):** Higher weight is given to PageRank (0.7) to prioritize popular pages.
   * **Long queries (>2 words):** Higher weight is given to BM25 (0.9) to prioritize textual relevance.

## How to Run

1. Install requirements:
   ```bash
   pip install flask google-cloud-storage pandas numpy
