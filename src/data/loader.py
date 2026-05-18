"""LineXRD data loader: 复用 ICL/xrd2c_data 的 NPY 格式。

数据格式: dict {features: (N, 8500) float32, labels230: (N,) int64}
位于 /opt/data/private/xrd2c_data/{dataset}_train/train_{dataset}.npy 等。
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATA_ROOT = Path("/opt/data/private/xrd2c_data")
XRD_LEN = 8500
N_CLASSES = 230  # space groups


SG_TO_CS = None  # 230 -> 7 crystal system mapping, lazy load


def _load_sg_to_cs(rule_matrix_path="/opt/data/private/ICL/rule_matrix.csv"):
    """从 rule_matrix.csv 加载 SG (1-230) -> CS (0-6) 映射."""
    import csv
    mapping = {}
    with open(rule_matrix_path) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                try:
                    sg, cs = int(row[0]), int(row[1])
                    mapping[sg] = cs
                except ValueError:
                    pass
    return mapping


def get_sg_to_cs():
    """Return deterministic SG (1..230) -> CS (0..6) mapping.

    The crystallographic ranges are fixed by the International Tables:
      triclinic    1-2,  monoclinic 3-15,  orthorhombic 16-74,
      tetragonal  75-142, trigonal  143-167, hexagonal  168-194,
      cubic       195-230.
    """
    global SG_TO_CS
    if SG_TO_CS is None or len(SG_TO_CS) == 0:
        cs_ranges = [(1, 2, 0), (3, 15, 1), (16, 74, 2), (75, 142, 3),
                     (143, 167, 4), (168, 194, 5), (195, 230, 6)]
        SG_TO_CS = {}
        for lo, hi, cs in cs_ranges:
            for sg in range(lo, hi + 1):
                SG_TO_CS[sg] = cs
    return SG_TO_CS


class PXRDDataset(Dataset):
    """ccdc_sg / inor / coremof19 共享格式."""

    def __init__(self, dataset: str, split: str):
        assert dataset in {"ccdc_sg", "inor", "coremof19"}
        assert split in {"train", "val", "test"}
        if split in {"train", "val"}:
            fname = "train" if split == "train" else "test_val"
            fp = DATA_ROOT / f"{dataset}_train" / f"{fname}_{dataset}.npy"
        else:
            fp = DATA_ROOT / f"{dataset}_test" / f"test_{dataset}.npy"
        d = np.load(fp, allow_pickle=True).item()
        self.features = d["features"]
        self.labels230 = d["labels230"]  # 0-indexed or 1-indexed depending on dataset
        # 统一改为 0-indexed (内部模型用)
        if self.labels230.min() == 1:
            self.labels230 = self.labels230 - 1
        sg_to_cs = get_sg_to_cs()
        self.labels_cs = np.array(
            [sg_to_cs.get(int(sg) + 1, 0) for sg in self.labels230], dtype=np.int64
        )

    def __len__(self):
        return len(self.labels230)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx]).float()  # (8500,)
        y_sg = torch.tensor(int(self.labels230[idx]), dtype=torch.long)
        y_cs = torch.tensor(int(self.labels_cs[idx]), dtype=torch.long)
        return x, y_sg, y_cs


def make_loaders(dataset: str, batch_size: int, num_workers: int = 4):
    train_ds = PXRDDataset(dataset, "train")
    val_ds = PXRDDataset(dataset, "val")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, train_ds, val_ds


if __name__ == "__main__":
    ds = PXRDDataset("ccdc_sg", "train")
    print("train size:", len(ds), "first x shape:", ds[0][0].shape, "y_sg:", ds[0][1].item())
