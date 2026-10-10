"""Deterministic real-data subset with exact first target timestamps."""
import argparse
import csv
import hashlib
import json
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description='Prepare the deterministic four-domain Books subset')
parser.add_argument('--ratings-dir', type=Path, default=Path('ratings-full'))
parser.add_argument('--out', type=Path, default=Path('artifacts/paper-four-domain'))
parser.add_argument('--seed', type=int, default=2026)
parser.add_argument('--limit', type=int, default=0, help='0 keeps all eligible users; 1000 reproduces the pilot')
args = parser.parse_args()
out = args.out
out.mkdir(parents=True, exist_ok=True)
raw = args.ratings_dir
seed = args.seed
sources = [('CDs', 'CDs_and_Vinyl.csv'), ('Movies', 'Movies_and_TV.csv'), ('Clothing', 'Clothing_Shoes_and_Jewelry.csv')]
source_events = defaultdict(dict)
start = time.monotonic()
for domain, filename in sources:
    total = 0
    with (raw / filename).open(newline='') as stream:
        for item, user, rating, timestamp in csv.reader(stream):
            total += 1
            if zlib.crc32(user.encode()) % 20:
                continue
            key = (domain, item)
            ts = int(timestamp) * 1000
            previous = source_events[user].get(key)
            if previous is None or ts < previous:
                source_events[user][key] = ts
    print(f'{domain}: scanned={total} candidate_users={len(source_events)} seconds={time.monotonic()-start:.1f}', flush=True)

target_events = defaultdict(dict)
with (raw / 'Books.csv').open(newline='') as stream:
    for index, (item, user, rating, timestamp) in enumerate(csv.reader(stream), 1):
        if user in source_events:
            ts = int(timestamp) * 1000
            previous = target_events[user].get(item)
            if previous is None or ts < previous:
                target_events[user][item] = ts
        if index % 10000000 == 0:
            print(f'Books: scanned={index} seconds={time.monotonic()-start:.1f}', flush=True)

eligible = []
excluded = Counter()
usable = {}
for user, items in target_events.items():
    if len(items) < 11:
        excluded['target_below_11'] += 1
        continue
    if len(items) > 100:
        excluded['target_above_100_small_run_limit'] += 1
        continue
    first = min(items.values())
    history = [(domain, item, ts) for (domain, item), ts in source_events[user].items() if ts < first]
    if not history:
        excluded['no_source_before_first_target'] += 1
        continue
    if max(Counter(domain for domain, _, _ in history).values()) > 100:
        excluded['source_above_100_small_run_limit'] += 1
        continue
    eligible.append(user)
    usable[user] = history
selected = sorted(eligible, key=lambda user: hashlib.sha256(f'{seed}:sample:{user}'.encode()).hexdigest())
if args.limit:
    selected = selected[:args.limit]
if len(selected) < 100:
    raise SystemExit(f'Too few eligible users: {len(selected)}')
counts = Counter()
with (out / 'events.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['user_id', 'domain', 'item_id', 'timestamp_ms'])
    for user in sorted(selected):
        for domain, item, ts in sorted(usable[user]):
            writer.writerow([user, domain, item, ts])
            counts[domain] += 1
        for item, ts in sorted(target_events[user].items(), key=lambda pair: (pair[1], pair[0])):
            writer.writerow([user, 'Books', item, ts])
            counts['Books'] += 1
summary = {'seed': seed, 'target': 'Books', 'sources': [d for d, _ in sources], 'user_hash_prefilter': 'crc32(user_id) % 20 == 0', 'eligible_pool': len(eligible), 'selected_users': len(selected), 'events_by_domain': dict(counts), 'excluded': dict(excluded), 'small_run_limits': '11-100 unique target items; at most 100 usable items per source domain; no item-core filtering', 'source_history': 'strictly before first target interaction', 'elapsed_seconds': time.monotonic()-start}
(out / 'subset.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
