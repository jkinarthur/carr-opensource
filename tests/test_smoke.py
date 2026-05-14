from carr_v2 import SyntheticInteractionDataset, make_loaders


def test_synthetic_dataset_len_and_item_shape() -> None:
    ds = SyntheticInteractionDataset(n_users=8, n_items=100, seq_len=12, seed=7)
    x, y = ds[0]
    assert len(ds) == 8
    assert x.shape[0] == 12
    assert y.ndim == 0


def test_make_loaders_splits_and_item_count() -> None:
    train_loader, val_loader, n_items = make_loaders(
        n_users=50,
        n_items=120,
        seq_len=10,
        val_frac=0.2,
        batch_size=8,
        num_workers=0,
        seed=1,
    )
    assert n_items == 120
    assert len(train_loader.dataset) + len(val_loader.dataset) == 50
