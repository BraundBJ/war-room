"""Insert data.json into the page template: python3 assemble.py WORK_DIR REPO_DIR"""
import os, sys
W, REPO = sys.argv[1], sys.argv[2]
s = open(f"{REPO}/site/page.src.html").read()
d = open(f"{W}/site/data.json").read().replace("</", "<\\/")
os.makedirs(f"{REPO}/public", exist_ok=True)
html = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">\n'
        + s.replace("__DATA__", d) + "\n</html>\n")
open(f"{REPO}/public/index.html", "w").write(html)
print("wrote public/index.html", round(len(html) / 1024), "KB")
