"""
file: test.py
author: Ale
description:
	- Evaluate DeepFM on a labeled test set with the same metrics as validation.
"""

import argparse
import logging
import os

import torch
import torch.utils.data as data_utils
from sklearn.metrics import (
	roc_auc_score,
	accuracy_score,
	log_loss,
	precision_score,
	recall_score,
	f1_score,
)
from tqdm import tqdm
from torch_rechub.models.ranking import DeepFM
from torch_rechub.basic.features import DenseFeature, SparseFeature

from dataset import CriteoDataset
from utils import seed_everything


def build_collate_fn(dense_features: list, sparse_features: list):
	def _collate(batch):
		labels = torch.stack([item[0] for item in batch])
		dense_x = torch.stack([item[1] for item in batch])
		sparse_x = torch.stack([item[2] for item in batch])

		x: dict[str, torch.Tensor] = {}
		for i, name in enumerate(dense_features):
			x[name] = dense_x[:, i]
		for i, name in enumerate(sparse_features):
			x[name] = sparse_x[:, i]

		return x, labels

	return _collate


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: data_utils.DataLoader, device: torch.device):
	model.eval()
	all_preds: list[float] = []
	all_labels: list[float] = []

	for x, labels in tqdm(loader, desc="  Eval ", leave=False, unit="batch"):
		x = {k: v.to(device) for k, v in x.items()}
		preds = model(x).cpu()
		all_preds.extend(preds.tolist())
		all_labels.extend(labels.tolist())

	auc = roc_auc_score(all_labels, all_preds)
	logloss = log_loss(all_labels, all_preds)

	bin_preds = [1 if p >= 0.5 else 0 for p in all_preds]
	acc = accuracy_score(all_labels, bin_preds)
	pre = precision_score(all_labels, bin_preds, zero_division=0)
	rec = recall_score(all_labels, bin_preds, zero_division=0)
	f1 = f1_score(all_labels, bin_preds, zero_division=0)

	return auc, logloss, acc, pre, rec, f1


def build_model(dense_features, sparse_features, vocab, embed_dim, mlp_dims, dropout):
	dense_feature_objs = [DenseFeature(name) for name in dense_features]
	sparse_feature_objs = [
		SparseFeature(name, vocab_size=vocab[name], embed_dim=embed_dim)
		for name in sparse_features
	]

	model = DeepFM(
		deep_features=dense_feature_objs + sparse_feature_objs,
		fm_features=sparse_feature_objs,
		mlp_params={
			"dims": mlp_dims,
			"dropout": dropout,
			"activation": "relu",
		},
	)
	return model


def _parse_mlp_dims(value: str) -> list[int]:
	return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
	parser = argparse.ArgumentParser(
		description="Evaluate DeepFM on a labeled test set using the same metrics as validation."
	)
	parser.add_argument("--data-path", required=True, help="Path to labeled test file")
	parser.add_argument(
		"--checkpoint",
		default=os.path.join("checkpoints", "deepfm_best.pth"),
		help="Path to model checkpoint",
	)
	parser.add_argument("--batch-size", type=int, default=1024)
	parser.add_argument("--embed-dim", type=int, default=32)
	parser.add_argument("--mlp-dims", type=_parse_mlp_dims, default="256,128,64")
	parser.add_argument("--dropout", type=float, default=0.3)
	parser.add_argument("--seed", type=int, default=42)
	return parser.parse_args()


def main():
	args = parse_args()
	seed_everything(args.seed)

	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)-8s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)
	logger = logging.getLogger(__name__)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logger.info("Device: %s", device)

	dataset = CriteoDataset(args.data_path)
	dense_feas = dataset.dense_features
	sparse_feas = dataset.sparse_features
	vocab = dataset.parse_vocab_size

	collate_fn = build_collate_fn(dense_feas, sparse_feas)
	loader = data_utils.DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=0,
		pin_memory=(device.type == "cuda"),
		collate_fn=collate_fn,
	)

	model = build_model(
		dense_feas,
		sparse_feas,
		vocab,
		embed_dim=args.embed_dim,
		mlp_dims=args.mlp_dims,
		dropout=args.dropout,
	).to(device)

	state = torch.load(args.checkpoint, map_location=device)
	model.load_state_dict(state)

	auc, logloss, acc, pre, rec, f1 = evaluate(model, loader, device)
	logger.info(
		"Test AUC: %.4f | LogLoss: %.4f | ACC: %.4f | Pre: %.4f | Rec: %.4f | F1: %.4f",
		auc,
		logloss,
		acc,
		pre,
		rec,
		f1,
	)


if __name__ == "__main__":
	main()