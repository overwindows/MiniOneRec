"""Check URA flight ID distribution in v8 data."""
import json
from collections import Counter

ura_flights = Counter()
total = 0
has_ura = 0

for split in ['train', 'dev']:
    path = f'data/doca_v8/{split}.jsonl'
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            total += 1
            fids = row.get('user_flight_ids', '') or ''
            ids = [x.strip() for x in fids.split(',') if x.strip()]
            matched = [x for x in ids if 'ura' in x.lower()]
            if matched:
                has_ura += 1
                for m in matched:
                    ura_flights[m] += 1

print(f'Total feeds: {total}')
print(f'Feeds with URA flight: {has_ura} ({has_ura/total*100:.2f}%)')
print(f'\nMatching flight IDs (containing "ura"):')
for fid, cnt in ura_flights.most_common():
    print(f'  {fid}: {cnt} ({cnt/total*100:.2f}%)')
