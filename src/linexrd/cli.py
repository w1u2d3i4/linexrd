"""Minimal training, reference-bank construction, calibration, and prediction."""
import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from . import fusion, verification
from .model import PeakSetTransformer
from .runtime import (destination, encode, integrity_error, load_encoder,
                      read_arrays, read_bank, retrieve, save_arrays)


def train(args):
    torch.manual_seed(args.seed)
    data, validation = read_arrays(args.train, True), read_arrays(args.validation, True)
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    config = dict(K=args.peaks, d_model=args.d_model, depth=args.depth,
                  n_heads=args.heads, bg_subtract=args.subtract_background)
    model = PeakSetTransformer(**config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loader = DataLoader(TensorDataset(torch.from_numpy(data['features']),
                                    torch.from_numpy(data['labels'])),
                        batch_size=args.batch_size, shuffle=True)
    counts = np.bincount(data['labels'], minlength=230)
    weight = 1 / np.sqrt(np.maximum(counts, 1))
    weight = torch.as_tensor(weight / weight.mean(), dtype=torch.float32, device=args.device)
    output = destination(args.checkpoint)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(args.device))['sg_logits']
            loss = torch.nn.functional.cross_entropy(logits, y.to(args.device),
                                                     weight=weight, label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        model.eval()
        _, p = encode(model, validation['features'], args.batch_size, args.device)
        score = float((p.argmax(1) == validation['labels']).mean())
        if score > best:
            best = score
            torch.save(dict(config=config, state_dict={name: tensor.detach().cpu()
                       for name, tensor in model.state_dict().items()}), output)
        print(f"epoch={epoch + 1} validation_top1={score:.6f}", flush=True)


def build_bank(args):
    data = read_arrays(args.data, True)
    model = load_encoder(args.checkpoint, args.device)
    z, _ = encode(model, data['features'], args.batch_size, args.device)
    metadata = {}
    for key in ('record_id', 'entity_id', 'source'):
        if key in data:
            if data[key].shape != (len(z),) or data[key].dtype.kind not in 'US':
                raise ValueError(f"{key} must contain one string per reference record")
            metadata[key] = data[key]
    save_arrays(args.output, embeddings=z, labels=data['labels'], **metadata)


def calibrate(args):
    data, bank = read_arrays(args.data, True), read_bank(args.bank)
    if len(data['features']) < 2:
        raise ValueError("Calibration requires at least two queries")
    model = load_encoder(args.checkpoint, args.device)
    z, p = encode(model, data['features'], args.batch_size, args.device)
    clean, rows = retrieve(z, p, bank, args.k, args.batch_size)
    clean['y'] = data['labels']
    # Mismatches are generated in memory from calibration queries only.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(z))
    cycle = np.arange(len(z))
    cycle[order] = np.roll(order, 1)
    wrong_rows = rows[cycle]
    actual = np.clip(np.einsum('nd,nkd->nk', z, bank['embeddings'][wrong_rows]), -1, 1)
    rank = np.argsort(-actual, axis=1, kind='stable')
    wrong = dict(p=p, sim=np.take_along_axis(actual, rank, axis=1),
                 neighbor_y=np.take_along_axis(bank['labels'][wrong_rows], rank, axis=1))
    conditional = fusion.fit(clean, args.k, args.temperature, args.seed,
                             args.tree_depth, args.tree_iterations,
                             args.mapping, args.benefit_threshold)
    verifier = verification.fit(clean, wrong, args.k, args.temperature, args.seed)
    bundle = dict(fusion=conditional, verifier=verifier,
                  relevance_threshold=args.relevance_threshold)
    with destination(args.output).open('wb') as stream:
        pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)


def predict(args):
    data = read_arrays(args.data)
    model = load_encoder(args.checkpoint, args.device)
    z, p = encode(model, data['features'], args.batch_size, args.device)
    output = dict(probabilities=p, space_group=p.argmax(1) + 1)
    if args.bank is not None:
        if args.calibration is None:
            raise ValueError("Retrieval prediction requires --calibration")
        bank = read_bank(args.bank)
        # Only load calibration artifacts you created or otherwise trust.
        with open(args.calibration, 'rb') as stream:
            bundle = pickle.load(stream)
        settings = bundle['fusion']
        payload, rows = retrieve(z, p, bank, settings['K'], args.batch_size)
        candidate, _, alpha = fusion.predict(settings, payload)
        error = integrity_error(z, bank, rows, payload['sim'])
        q, accepted = verification.apply(bundle['verifier'], payload, candidate,
                                         bundle['relevance_threshold'], error)
        output.update(probabilities=q, space_group=q.argmax(1) + 1,
                      accepted=accepted, alpha=alpha, reference_row=rows,
                      similarities=payload['sim'])
        for key in ('record_id', 'entity_id', 'source'):
            if key in bank:
                output[key] = bank[key][rows]
    elif args.calibration is not None:
        raise ValueError("--calibration requires --bank")
    save_arrays(args.output, **output)


def optional_float(text):
    return None if text.lower() == 'off' else float(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train', help='Fit a PST on user-supplied arrays')
    training.add_argument('--train', required=True)
    training.add_argument('--validation', required=True)
    training.add_argument('--checkpoint', required=True)
    training.add_argument('--epochs', type=int, default=60)
    training.add_argument('--lr', type=float, default=2e-4)
    training.add_argument('--peaks', type=int, default=512)
    training.add_argument('--d-model', type=int, default=256)
    training.add_argument('--depth', type=int, default=6)
    training.add_argument('--heads', type=int, default=8)
    training.add_argument('--subtract-background', action='store_true')
    training.add_argument('--seed', type=int, default=0)
    training.set_defaults(function=train)
    for name, function in [('build-bank', build_bank), ('calibrate', calibrate), ('predict', predict)]:
        command = commands.add_parser(name)
        command.add_argument('--data', required=True)
        command.add_argument('--checkpoint', required=True)
        command.add_argument('--output', required=True)
        command.set_defaults(function=function)
        if name == 'calibrate':
            command.add_argument('--bank', required=True)
            command.add_argument('--k', type=int, default=4)
            command.add_argument('--temperature', type=float, default=0.03)
            command.add_argument('--seed', type=int, default=0)
            command.add_argument('--tree-depth', type=int, default=3)
            command.add_argument('--tree-iterations', type=int, default=200)
            command.add_argument('--mapping', choices=['hard', 'soft'], default='hard')
            command.add_argument('--benefit-threshold', type=optional_float, default=0.1)
            command.add_argument('--relevance-threshold', type=float, default=0.1)
        elif name == 'predict':
            command.add_argument('--bank')
            command.add_argument('--calibration')
    for command in commands.choices.values():
        command.add_argument('--batch-size', type=int, default=64)
        command.add_argument('--device', default='cpu')
    args = parser.parse_args()
    args.function(args)
