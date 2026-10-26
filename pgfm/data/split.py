def split_by_day(all_days: list[str], split_day: str):
    train_days = [d for d in all_days if d < split_day]
    val_days   = [d for d in all_days if d >= split_day]
    if not train_days or not val_days:
        raise RuntimeError("Bad split_day: ensure both train and val are non-empty.")
    return train_days, val_days


class SplitDays(list):
    """Source days plus allowed UTC origin dates; padding never adds origins."""
    def __init__(self, days, origin_days):
        super().__init__(days)
        self.origin_days = list(origin_days)
    def __getitem__(self, index):
        if isinstance(index, slice):
            raise ValueError("Slice origin dates explicitly; do not discard split metadata")
        return super().__getitem__(index)


def configured_splits(all_days, data_cfg):
    rule = data_cfg.get("monthly_origin_split")
    if not rule:
        train, val = split_by_day(all_days, str(data_cfg["split_day"]))
        return {"train": train, "val": val}
    if int(data_cfg.get("past_hours", 24)) != 24 or int(data_cfg.get("horizon_hours", 24)) != 24:
        raise ValueError("Monthly buffered split currently requires 24h history/horizon")
    from datetime import date, timedelta
    available = set(all_days)
    months = sorted({d[:7] for d in all_days})
    blocks = {}; held = set()
    for split, count in (("val", 2), ("test", 3)):
        numbers = rule[split]
        if len(numbers) != count or len(set(numbers)) != count:
            raise ValueError(f"{split} requires exactly {count} distinct origin days per month")
        origins = [date.fromisoformat(f"{month}-{int(n):02d}").isoformat() for month in months for n in numbers]
        source = set()
        for d in origins:
            origin = date.fromisoformat(d)
            # Full calendar-day buffer; both 24h history and 24h future remain isolated.
            for delta in (-1, 0, 1): source.add((origin + timedelta(days=delta)).isoformat())
        missing = source - available
        if missing: raise ValueError(f"Missing {split} origin/buffer source dates: {sorted(missing)}")
        if source & held: raise ValueError("Validation and test buffers overlap")
        held |= source
        blocks[split] = SplitDays(sorted(source), origins)
    train = sorted(available-held)
    blocks["train"] = SplitDays(train, train)
    return blocks
