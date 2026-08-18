import argparse
import json
import os
import re
import time
import xml.etree.ElementTree as ET

import requests

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"
ATOM_NAMESPACE = "{http://www.w3.org/2005/Atom}"


def parse_args():
    parser = argparse.ArgumentParser(description="Download a small arXiv PDF sample")

    parser.add_argument("--num_papers", type=int, default=25,
                         help="Number of arXiv papers to download")
    parser.add_argument("--category", type=str, default="cs.CL",
                         help="arXiv category to search within")
    parser.add_argument("--output_dir", type=str, default="data/raw/pdf_sample",
                         help="Directory to write the sampled .pdf files into")

    return parser.parse_args()


def sanitize_title(title):
    single_line_title = " ".join(title.split())
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", single_line_title).strip("_")
    return sanitized.lower()


def build_filename(title, arxiv_id):
    sanitized_title = sanitize_title(title)
    sanitized_id = arxiv_id.replace(".", "_")
    return f"{sanitized_title}_{sanitized_id}.pdf"


def fetch_arxiv_entries(category, num_papers):
    query_params = {
        "search_query": f"cat:{category}",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": 0,
        "max_results": num_papers,
    }
    response = requests.get(ARXIV_API_URL, params=query_params, timeout=30)
    response.raise_for_status()

    feed = ET.fromstring(response.text)
    entries = []
    for entry_element in feed.findall(f"{ATOM_NAMESPACE}entry"):
        entry_id_url = entry_element.find(f"{ATOM_NAMESPACE}id").text.strip()
        arxiv_id = entry_id_url.rsplit("/", 1)[-1]
        title = entry_element.find(f"{ATOM_NAMESPACE}title").text.strip()
        entries.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "url": entry_id_url,
        })

    return entries


def download_pdf_sample(num_papers, category, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    entries = fetch_arxiv_entries(category, num_papers)

    filename_to_sidecar_metadata = {}
    num_papers_downloaded = 0
    for entry in entries:
        pdf_response = requests.get(ARXIV_PDF_URL.format(arxiv_id=entry["arxiv_id"]), timeout=60)
        pdf_response.raise_for_status()

        filename = build_filename(entry["title"], entry["arxiv_id"])
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "wb") as output_file:
            output_file.write(pdf_response.content)

        filename_to_sidecar_metadata[filename] = {
            "arxiv_id": entry["arxiv_id"],
            "title": entry["title"],
            "url": entry["url"],
        }

        print(f"Wrote {output_path}")
        num_papers_downloaded += 1
        time.sleep(1)

    sidecar_path = os.path.join(output_dir, "metadata.json")
    with open(sidecar_path, "w", encoding="utf-8") as sidecar_file:
        json.dump(filename_to_sidecar_metadata, sidecar_file, indent=2)

    print(f"Downloaded {num_papers_downloaded} papers to {output_dir}")
    print(f"Wrote sidecar metadata to {sidecar_path}")


if __name__ == "__main__":
    args = parse_args()
    download_pdf_sample(args.num_papers, args.category, args.output_dir)
