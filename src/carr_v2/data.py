from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticInteractionDataset(Dataset):
    """
    Synthesizes sequential interaction data with multi-intent patterns at scale.

    Items are partitioned into n_intents groups; users sample from 1-2 dominant
    clusters, creating realistic multi-intent session histories.
    Designed for publication-grade reproducible experiments at 100k-user scale.
    """

    def __init__(
        self,
        n_users: int = 100_000,
        n_items: int = 20_000,
        n_intents: int = 8,
        seq_len: int = 50,
        seed: int = 42,
    ):
        super().__init__()
        rng = random.Random(seed)
        torch.manual_seed(seed)

        intent_size = n_items // n_intents
        self.sequences: list[torch.Tensor] = []
        self.targets: list[int] = []
        for _ in range(n_users):
            primary = rng.randint(0, n_intents - 1)
            secondary = rng.randint(0, n_intents - 1)
            pool = (
                list(range(primary * intent_size, (primary + 1) * intent_size))
                + list(range(secondary * intent_size, (secondary + 1) * intent_size))
            )
            pool = [x % n_items + 1 for x in pool]
            history = rng.choices(pool, k=seq_len + 1)
            self.sequences.append(torch.tensor(history[:seq_len], dtype=torch.long))
            self.targets.append(history[seq_len] - 1)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], torch.tensor(self.targets[idx], dtype=torch.long)


class RealInteractionDataset(Dataset):
    """
    Loads real interaction data from a tab/comma-separated file.

    Expected columns (detected via header): user_id, item_id, timestamp.
    Sequences are built from the most recent seq_len+1 interactions per user;
    the final item becomes the prediction target. Compatible with MovieLens,
    Amazon review exports, and similar benchmark formats.

    Args:
        file_path: Path to .tsv / .csv interaction file.
        seq_len:   Input sequence length (default 50).
        min_interactions: Drop users with fewer interactions than this.
        sep:       Column separator (default tab).
    """

    def __init__(
        self,
        file_path: str,
        seq_len: int = 50,
        min_interactions: int = 10,
        sep: str = "\t",
    ):
        import csv as _csv

        super().__init__()
        raw: dict[int, list[tuple[int, int]]] = {}  # user_id -> [(item_idx, ts)]
        item_map: dict[str, int] = {}

        with open(file_path, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f, delimiter=sep)
            for row in reader:
                uid = int(row["user_id"])
                iid_raw = row["item_id"]
                ts = int(row.get("timestamp", 0))
                if iid_raw not in item_map:
                    item_map[iid_raw] = len(item_map) + 1  # 1-indexed
                raw.setdefault(uid, []).append((item_map[iid_raw], ts))

        self.num_items = len(item_map)
        self.sequences: list[torch.Tensor] = []
        self.targets: list[int] = []
        for interactions in raw.values():
            if len(interactions) < min_interactions:
                continue
            interactions.sort(key=lambda x: x[1])
            items = [i for i, _ in interactions]
            window = items[-(seq_len + 1):]
            seq = window[:seq_len]
            target = window[-1] - 1  # 0-indexed
            if len(seq) < seq_len:
                seq = [0] * (seq_len - len(seq)) + seq  # left-pad with zeros
            self.sequences.append(torch.tensor(seq, dtype=torch.long))
            self.targets.append(target)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], torch.tensor(self.targets[idx], dtype=torch.long)


def make_loaders(
    n_users: int = 100_000,
    n_items: int = 20_000,
    seq_len: int = 50,
    val_frac: float = 0.10,
    batch_size: int = 512,
    num_workers: int = 4,
    seed: int = 42,
    data_path: str | None = None,
) -> tuple[DataLoader, DataLoader, int]:
    """
    Build train / validation DataLoaders.

    Args:
        data_path: If given, loads real data via RealInteractionDataset;
                   otherwise synthesizes data with SyntheticInteractionDataset.
        num_workers: Set to 0 on Windows if multiprocessing errors occur.
    Returns:
        (train_loader, val_loader, n_items)
    """
    if data_path is not None:
        dataset: Dataset = RealInteractionDataset(data_path, seq_len=seq_len)
        n_items = dataset.num_items  # type: ignore[attr-defined]
    else:
        dataset = SyntheticInteractionDataset(
            n_users=n_users, n_items=n_items, seq_len=seq_len, seed=seed
        )

    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    pin = torch.cuda.is_available()
    persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=persistent,
    )
    return train_loader, val_loader, n_items
