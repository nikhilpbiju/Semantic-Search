import gzip
import json
import pandas as pd

data = []

with gzip.open("/Users/nikhil/Documents/semantic-search/data/meta_Electronics.jsonl.gz", "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i > 50000:   # limit for speed
            break

        item = json.loads(line)

        title = item.get("title", "")
        desc = " ".join(item.get("description", []))

        if title and desc and len(title) > 5:
            text = (title + " " + desc).strip()

            data.append({
                "title": title.strip(),
                "description": desc.strip(),
                "text": text
            })

df = pd.DataFrame(data)

# cleaning
df = df.drop_duplicates(subset=["text"])
df = df[df["text"].str.len() > 20]

df.to_csv("data/products.csv", index=False)

print("Saved:", len(df))