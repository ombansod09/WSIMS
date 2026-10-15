"""Pure helpers for dashboard summaries, trends, and class filters."""
from collections import Counter
from datetime import datetime

def summarize(records, class_names):
    counts = Counter(row["predicted_class"] for row in records)
    return {
        "total": len(records),
        "class_counts": {name: int(counts.get(name, 0)) for name in class_names},
        "records": records,
    }

def daily_counts(records):
    counts = Counter()
    for row in records:
        stamp = row.get("timestamp_utc", "")
        day = stamp[:10]
        if day:
            counts[day] += 1
    return [{"date": day, "predictions": counts[day]} for day in sorted(counts)]
