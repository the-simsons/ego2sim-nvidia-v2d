"""List a public Google Drive folder tree (names, ids, sizes) WITHOUT downloading any file.

  python tools/list_gdrive.py FOLDER_URL [--out list.tsv]

Folder structure via gdown (skip_download). Size per file from the HTTP headers / virus-scan page of the download
endpoint: only the first bytes are requested (Range) and the connection is closed, so no file content is kept.
"""
import argparse
import re
import sys

import gdown
import requests


def file_size(fid, s):
    r = s.get("https://drive.usercontent.google.com/download", params={"id": fid, "export": "download"},
              headers={"Range": "bytes=0-0"}, stream=True, timeout=30)
    try:
        cr = r.headers.get("Content-Range", "")
        if cr and "/" in cr:
            return int(cr.split("/")[-1]), "bytes"
        if "text/html" in r.headers.get("Content-Type", ""):
            html = r.raw.read(200000).decode("utf-8", "ignore")
            m = re.search(r"\(([\d.]+)\s*([KMGT])B?\)", html)
            if m:
                mult = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}[m.group(2)]
                return int(float(m.group(1)) * mult), "approx"
            if "quota" in html.lower() or "sign in" in html.lower():
                return -1, "quota/sign-in"
            return -1, "unknown"
        return int(r.headers.get("Content-Length", -1)), "content-length"
    finally:
        r.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    files = gdown.download_folder(a.url, skip_download=True, quiet=True)
    s = requests.Session()
    out = open(a.out, "w") if a.out else sys.stdout
    print("path\tid\tsize_bytes\tsize_source", file=out)
    for f in files:
        sz, src = file_size(f.id, s)
        print(f"{f.path}\t{f.id}\t{sz}\t{src}", file=out, flush=True)


if __name__ == "__main__":
    main()
