"""Deterministic manifest holdout views shared by bounded evaluators."""

import random


def selected_views(manifest: dict, cap: int, seed: int) -> dict[str, list[int]]:
    if cap < 1 or seed < 0:
        raise ValueError("invalid validation selection")
    global_indices = list(manifest["validation"]["indices"])
    random.Random(seed).shuffle(global_indices)
    views = {"global": global_indices[:cap]}
    for offset, (domain, indices) in enumerate(sorted(
            manifest["validation"]["domain_indices"].items())):
        selected = list(indices)
        random.Random(seed + offset + 1).shuffle(selected)
        if selected:
            views[f"domain:{domain}"] = selected[:cap]
    for client_id, entry in sorted(manifest["validation"]["clients"].items()):
        selected = list(entry["indices"])
        random.Random(seed + int(client_id) + 1001).shuffle(selected)
        if selected:
            views[f"client:{client_id}"] = selected[:cap]
    return views
