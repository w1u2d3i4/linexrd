"""Local array I/O, encoder loading, and bounded-memory cosine retrieval."""
from pathlib import Path
import numpy as np
import torch
from .model import L_INPUT, PeakSetTransformer


def read_arrays(path, labeled=False):
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    x = np.asarray(arrays['features'], dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != L_INPUT or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("features must be a nonempty finite (N, 8500) array")
    arrays['features'] = x
    if labeled:
        y = np.asarray(arrays['labels'])
        if y.shape != (len(x),) or y.dtype.kind not in 'iu' or ((y < 0) | (y >= 230)).any():
            raise ValueError("labels must be (N,) integers in [0, 229]")
        arrays['labels'] = y.astype(np.int64)
    return arrays


def destination(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Choose a new output path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_arrays(path, **arrays):
    with destination(path).open('wb') as stream:
        np.savez_compressed(stream, **arrays)


def load_encoder(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = PeakSetTransformer(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device).eval()


@torch.inference_mode()
def encode(model, x, batch_size, device):
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    embeddings, probabilities = [], []
    for start in range(0, len(x), batch_size):
        h = model.encode(torch.from_numpy(x[start:start + batch_size]).to(device))
        p = model.cls_sg(h).softmax(-1)
        h = torch.nn.functional.normalize(h, dim=-1)
        embeddings.append(h.cpu().numpy())
        probabilities.append(p.cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(probabilities)


def read_bank(path):
    with np.load(path, allow_pickle=False) as data:
        bank = {name: data[name] for name in data.files}
    z, y = bank['embeddings'], bank['labels']
    if z.ndim != 2 or not len(z) or not np.isfinite(z).all():
        raise ValueError("Invalid bank embeddings")
    if y.shape != (len(z),) or y.dtype.kind not in 'iu' or ((y < 0) | (y >= 230)).any():
        raise ValueError("Invalid bank labels")
    if not np.allclose(np.linalg.norm(z, axis=1), 1, atol=1e-4):
        raise ValueError("Bank embeddings must be normalized")
    for name in ('record_id', 'entity_id', 'source'):
        if name in bank and bank[name].shape != (len(z),):
            raise ValueError(f"Invalid bank metadata: {name}")
    return bank


def retrieve(z, p, bank, k, batch_size):
    reference = bank['embeddings']
    if not 1 <= k <= len(reference) or z.shape[1] != reference.shape[1] or batch_size < 1:
        raise ValueError("Invalid k, batch size, or bank embedding dimension")
    scores, indices = [], []
    for start in range(0, len(z), batch_size):
        similarity = np.clip(z[start:start + batch_size] @ reference.T, -1, 1)
        top = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        values = np.take_along_axis(similarity, top, axis=1)
        order = np.argsort(-values, axis=1, kind='stable')
        scores.append(np.take_along_axis(values, order, axis=1))
        indices.append(np.take_along_axis(top, order, axis=1))
    rows = np.concatenate(indices)
    payload = dict(p=p, sim=np.concatenate(scores), neighbor_y=bank['labels'][rows])
    return payload, rows


def integrity_error(z, bank, rows, reported):
    actual = np.einsum('nd,nkd->nk', z, bank['embeddings'][rows])
    return np.max(np.abs(actual - reported), axis=1)
