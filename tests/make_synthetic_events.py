import csv
from pathlib import Path

output = Path("artifacts/synthetic/events.csv")
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(("user_id", "domain", "item_id", "timestamp_ms"))
    for user in range(100):
        writer.writerow((f"u{user}", "CDs", f"s{user % 30}", 0))
        for index in range(12):
            writer.writerow((f"u{user}", "Books", f"t{(user * 13 + index) % 200}", 1000000 + index))
