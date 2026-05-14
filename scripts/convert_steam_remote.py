from pathlib import Path
import gzip
import ast
from datetime import datetime

inp = Path('/home/ubuntu/carr-opensource/data/raw/steam_reviews.json.gz')
out = Path('/home/ubuntu/carr-opensource/data/datasets/Steam/interactions.tsv')
out.parent.mkdir(parents=True, exist_ok=True)

count = 0
with gzip.open(inp, 'rt', encoding='utf-8', errors='ignore') as f, out.open('w', encoding='utf-8') as w:
    w.write('user_id\titem_id\ttimestamp\n')
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = ast.literal_eval(line)
        except Exception:
            continue

        user = d.get('username') or d.get('user_id') or d.get('reviewerID')
        item = d.get('product_id') or d.get('item_id') or d.get('asin')
        ts = d.get('timestamp') or d.get('unixReviewTime')
        if ts is None:
            date_str = d.get('date')
            if date_str:
                try:
                    ts = int(datetime.strptime(str(date_str), '%Y-%m-%d').timestamp())
                except Exception:
                    ts = 0
            else:
                ts = 0
        if user and item:
            try:
                ts_int = int(ts)
            except Exception:
                ts_int = 0
            w.write(f"{user}\t{item}\t{ts_int}\n")
            count += 1

print(f'steam_rows={count}')
