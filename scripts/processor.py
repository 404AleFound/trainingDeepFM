"""
file: data.py
author: Ale
date: 2026/04/19

说明：
    - 读取 Criteo 原始 txt（制表符分隔），先转换为 csv。
    - 后续所有处理均基于 csv 文件。
    - 缺失值、稠密特征、稀疏特征采用可插拔（多态）处理策略。
"""

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import tqdm

# Criteo 列名
CONTINUOUS_COLUMNS = ["I" + str(i) for i in range(1, 14)]
CATEGORICAL_COLUMNS = ["C" + str(i) for i in range(1, 27)]
LABEL_COLUMN = "label"
TRAIN_DATA_COLUMNS = [LABEL_COLUMN] + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
CSV_HEADER_COLUMNS = [
    LABEL_COLUMN,
    *["i" + str(i) for i in range(1, 14)],
    *["c" + str(i) for i in range(1, 27)],
]


@dataclass
class NumericStats:
    """数值特征统计配置。

    示例:
        stats = NumericStats(avg=3.6, stddev=14.5, bucketize_values=[0.0, 1.0, 2.0])
    """

    avg: float
    stddev: float
    bucketize_values: List[float]


# 数值特征预处理参数
NUMERIC_FEATURE_STATS: Dict[str, NumericStats] = {
    "I1": NumericStats(3.6, 14.5, [0.0, 1.0, 2.0, 3.0, 5.0, 9.0]),
    "I2": NumericStats(86.7, 328.6, [-1.0, 0.0, 1.0, 2.0, 7.0, 19.0, 48.0, 163.0]),
    "I3": NumericStats(26.1, 493.4, [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 16.0, 30.0]),
    "I4": NumericStats(6.6, 8.0, []),
    "I5": NumericStats(
        17917.2,
        67045.7,
        [14.0, 691.0, 1051.0, 1402.0, 2700.0, 5325.0, 7670.0, 12915.0, 34205.0],
    ),
    "I6": NumericStats(111.7, 350.1, [1.0, 5.0, 11.0, 19.0, 31.0, 49.0, 77.0, 132.0, 265.0]),
    "I7": NumericStats(16.6, 69.2, [0.0, 1.0, 2.0, 4.0, 6.0, 9.0, 16.0, 34.0]),
    "I8": NumericStats(13.3, 17.5, [0.0, 2.0, 3.0, 5.0, 8.0, 12.0, 17.0, 25.0, 36.0]),
    "I9": NumericStats(106.6, 210.0, [3.0, 8.0, 15.0, 26.0, 41.0, 61.0, 92.0, 147.0, 268.0]),
    "I10": NumericStats(0.6, 0.7, [0.0, 1.0]),
    "I11": NumericStats(2.9, 5.4, [0.0, 1.0, 2.0, 4.0, 7.0]),
    "I12": NumericStats(1.0, 6.2, [0.0, 1.0]),
    "I13": NumericStats(6.5, 15.2, [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 16.0]),
}

# 类别特征桶大小（哈希或截断词表的上限）
CATEGORY_FEATURE_STATS: Dict[str, int] = {
    "C1": 1036,
    "C2": 530,
    "C3": 169550,
    "C4": 71524,
    "C5": 241,
    "C6": 15,
    "C7": 10025,
    "C8": 458,
    "C9": 3,
    "C10": 22960,
    "C11": 4469,
    "C12": 144780,
    "C13": 3034,
    "C14": 26,
    "C15": 7577,
    "C16": 113860,
    "C17": 10,
    "C18": 3440,
    "C19": 1678,
    "C20": 3,
    "C21": 130892,
    "C22": 11,
    "C23": 14,
    "C24": 27189,
    "C25": 65,
    "C26": 20188,
}


class ProcessorBase(object):
    """特征处理基类"""
    def __init__(self) -> None:
        self.vocab_size = 0

    def fit(self, series: pd.Series) -> None:
        pass

    def transform(self, series: pd.Series) -> np.ndarray:
        raise NotImplementedError


class HashBucketProcessor(ProcessorBase):
    """固定桶大小的哈希映射（无需拟合词表）。"""
    def __init__(self, num_bins: int) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.vocab_size = num_bins

    def transform(self, series: pd.Series) -> np.ndarray:
        return series.fillna("unknown").astype(str).map(
            lambda x: int.from_bytes(hashlib.md5(str(x).encode("utf-8")).digest(), "big") % self.num_bins
        ).astype(np.int64).to_numpy()


class LookupTableProcessor(ProcessorBase):
    """基于训练集词表映射，未登录值映射到 __UNK__。"""
    def __init__(self, max_bins: Optional[int] = None) -> None:
        super().__init__()
        self.max_bins = max_bins
        self.mapping: Dict[str, int] = {}

    def fit(self, series: pd.Series) -> None:
        uniques = series.fillna("unknown").astype(str).unique()
        if self.max_bins is not None:
            uniques = uniques[: max(0, self.max_bins - 1)]
        self.mapping = {val: idx for idx, val in enumerate(uniques)}
        self.mapping["__UNK__"] = len(self.mapping)
        self.vocab_size = len(self.mapping)

    def transform(self, series: pd.Series) -> np.ndarray:
        unk = self.mapping.get("__UNK__", 0)
        return series.fillna("unknown").astype(str).map(self.mapping).fillna(unk).astype(np.int64).to_numpy()


class StatsStandardizeProcessor(ProcessorBase):
    """使用均值/方差做标准化。如果在配置中缺失，则在线计算。"""
    def __init__(self, stats: NumericStats) -> None:
        super().__init__()
        self.avg = stats.avg
        self.stddev = stats.stddev

    def fit(self, series: pd.Series) -> None:
        if self.avg is None or self.stddev is None:
            vals = series.dropna().astype(float)
            self.avg, self.stddev = (float(vals.mean()), float(vals.std())) if len(vals) > 0 else (0.0, 1.0)

    def transform(self, series: pd.Series) -> np.ndarray:
        values = series.astype(float).to_numpy()
        return ((values - self.avg) / (self.stddev if self.stddev else 1.0)).astype(np.float32)


class BaseBucketProcessor(ProcessorBase):
    """分桶处理核心逻辑，支持归一化或直接输出离散索引。"""
    def __init__(self, stats: NumericStats, num_bins: int, max_bins: int, normalize: bool) -> None:
        super().__init__()
        self.num_bins, self.max_bins, self.normalize = num_bins, max_bins, normalize
        if len(stats.bucketize_values) > 0:
            self.bins = np.array(stats.bucketize_values, dtype=float)
            if len(self.bins) > self.max_bins:
                self.bins = self.bins[np.linspace(0, len(self.bins) - 1, self.max_bins).astype(int)]
            self.vocab_size = len(self.bins) + 1
        else:
            self.bins = np.array([], dtype=float)
            self.vocab_size = min(num_bins, max_bins) + 1

    def fit(self, series: pd.Series) -> None:
        if len(self.bins) == 0:
            vals = series.dropna().astype(float)
            if len(vals) > 0:
                actual_bins = min(self.num_bins, self.max_bins)
                pcts = np.linspace(0, 100, actual_bins + 1)[1:-1]
                self.bins = np.unique(np.percentile(vals, pcts))
            else:
                self.bins = np.array([0.0], dtype=float)
            self.vocab_size = len(self.bins) + 1

    def transform(self, series: pd.Series) -> np.ndarray:
        binned = np.digitize(series.astype(float).to_numpy(), self.bins, right=False)
        if self.normalize:
            return (binned / max(len(self.bins), 1)).astype(np.float32)
        return binned.astype(np.int64)


class StatsBucketizeProcessor(BaseBucketProcessor):
    """使用分桶边界进行归一化。"""
    def __init__(self, stats: NumericStats, num_bins: int = 10, max_bins: int = 20) -> None:
        super().__init__(stats, num_bins, max_bins, normalize=True)


class DenseToSparseBucketProcessor(BaseBucketProcessor):
    """把稠密特征分桶并转为离散索引（当作类别特征），不再归一化。"""
    def __init__(self, stats: NumericStats, num_bins: int = 10, max_bins: int = 20) -> None:
        super().__init__(stats, num_bins, max_bins, normalize=False)

class MinMaxFitProcessor(ProcessorBase):
    """在训练数据上拟合 min/max，再做归一化(基于 sklearn)"""
    def __init__(self) -> None:
        super().__init__()
        self.scaler = MinMaxScaler()

    def fit(self, series: pd.Series) -> None:
        vals = series.dropna().astype(float).to_numpy().reshape(-1, 1)
        if len(vals) > 0:
            self.scaler.fit(vals)

    def transform(self, series: pd.Series) -> np.ndarray:
        vals = series.astype(float).to_numpy().reshape(-1, 1)
        if not hasattr(self.scaler, 'data_min_'): 
            return vals.reshape(-1).astype(np.float32)
        return self.scaler.transform(vals).reshape(-1).astype(np.float32)


def build_dense_processors(mode: str = "stats") -> Dict[str, ProcessorBase]:
    """
    构建稠密特征处理器字典。

    mode:
        - "stats" : 优先分桶，若无分桶则使用均值/方差标准化
        - "minmax": 基于训练集拟合 min/max

    示例:
        输入:
            mode="stats" 或 mode="minmax"
        输出:
            {"I1":processor, ..., "I13":processor}
        原因解释:
            用策略工厂统一构建处理器，后续切换方案无需改主流程。
    """
    processors: Dict[str, ProcessorBase] = {}
    if mode == "stats":
        for col in CONTINUOUS_COLUMNS:
            stats = NUMERIC_FEATURE_STATS[col]
            if stats.bucketize_values:
                processors[col] = StatsBucketizeProcessor(stats)
            else:
                processors[col] = StatsStandardizeProcessor(stats)
    elif mode == "minmax":
        for col in CONTINUOUS_COLUMNS:
            processors[col] = MinMaxFitProcessor()
    elif mode == "bucketize_as_sparse":
        for col in CONTINUOUS_COLUMNS:
            stats = NUMERIC_FEATURE_STATS[col]
            processors[col] = DenseToSparseBucketProcessor(stats)
    else:
        raise ValueError(f"Unknown dense processor mode: {mode}")
    return processors


def build_sparse_processors(mode: str = "hash") -> Dict[str, ProcessorBase]:
    """
    构建稀疏特征处理器字典。

    mode:
        - "hash"  : 固定桶大小哈希（默认）
        - "lookup": 训练集词表映射（可设置 max_bins）

    示例:
        输入:
            mode="hash" 或 mode="lookup"
        输出:
            {"C1":processor, ..., "C26":processor}
        原因解释:
            用多态封装不同稀疏编码策略，便于做横向实验对比。
    """
    processors: Dict[str, ProcessorBase] = {}
    if mode == "hash":
        for col in CATEGORICAL_COLUMNS:
            processors[col] = HashBucketProcessor(CATEGORY_FEATURE_STATS[col])
    elif mode == "lookup":
        for col in CATEGORICAL_COLUMNS:
            processors[col] = LookupTableProcessor(max_bins=CATEGORY_FEATURE_STATS[col])
    else:
        raise ValueError(f"Unknown sparse processor mode: {mode}")
    return processors


def _to_csv_path(data_path: str) -> str:
    """把输入路径转换为对应 csv 路径。

    示例:
        输入:
            "data/train.txt" 或 "data/train.csv"
        输出:
            "data/train.csv"
        原因解释:
            统一后缀，避免后续读取逻辑分叉。
    """
    base, ext = os.path.splitext(data_path)
    if ext.lower() == ".csv":
        return data_path
    return base + ".csv"


def ensure_csv(data_path: str) -> str:
    """如果没有 csv，就先从原始 txt 转成 csv。

    示例:
        输入:
            "./data/Criteo/train.txt"
        输出:
            "./data/Criteo/train.csv"（若已存在则直接返回）
        原因解释:
            先标准化为 csv，后续所有处理都走同一格式，提升可维护性。
    """
    csv_path = _to_csv_path(data_path)
    if os.path.exists(csv_path):
        # 已存在 csv 时，若表头不合规则尝试重建，避免后续按列名读取失败
        try:
            head_df = pd.read_csv(csv_path, nrows=0)
            cols = [str(c).strip() for c in head_df.columns.tolist()]
            valid_cols = set(CSV_HEADER_COLUMNS) | set(TRAIN_DATA_COLUMNS)
            if len(cols) == len(CSV_HEADER_COLUMNS) and all(c in valid_cols for c in cols):
                return csv_path
        except Exception:
            pass
        print(f"Existing csv header is invalid, rebuilding: {csv_path}")

    # 原始 txt 转 csv，并写入标准表头（label,i1..i13,c1..c26）
    if os.path.exists(csv_path):
        os.remove(csv_path)

    print(f"Converting raw txt to csv: {csv_path}")
    chunk_size = 500000
    df_iter = pd.read_csv(
        data_path,
        sep="\t",
        header=None,
        names=TRAIN_DATA_COLUMNS,
        chunksize=chunk_size,
    )
    for idx, chunk in enumerate(df_iter):
        chunk.columns = CSV_HEADER_COLUMNS
        mode = "w" if idx == 0 else "a"
        header = idx == 0
        chunk.to_csv(csv_path, index=False, mode=mode, header=header)

    return csv_path


class CriteoDataProcessor(object):
    """
    数据处理流程封装（多态）：
    - 确保 csv 存在
    - 缺失值处理
    - 稠密/稀疏特征处理

    示例:
        输入:
            dense_mode="stats", sparse_mode="hash", data_path="./data/Criteo/train.txt"
        输出:
            处理后的 DataFrame（label + 13 稠密 + 26 稀疏）
        原因解释:
            把文件读取、缺失值处理、特征编码统一收口，便于复用与对比实验。
    """

    def __init__(
        self,
        dense_mode: str = "stats",
        sparse_mode: str = "hash",
        fill_dense: float = 0.0,
        fill_sparse: str = "unknown",
    ) -> None:
        self.dense_mode = dense_mode
        self.sparse_mode = sparse_mode
        self.fill_dense = fill_dense
        self.fill_sparse = fill_sparse
        self.dense_processors = build_dense_processors(dense_mode)
        self.sparse_processors = build_sparse_processors(sparse_mode)
        self._is_fitted = False

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """将 csv 表头统一映射为内部使用列名（label/I1..I13/C1..C26）。"""
        rename_map = {"label": "label"}
        rename_map.update({f"i{i}": f"I{i}" for i in range(1, 14)})
        rename_map.update({f"c{i}": f"C{i}" for i in range(1, 27)})
        rename_map.update({f"I{i}": f"I{i}" for i in range(1, 14)})
        rename_map.update({f"C{i}": f"C{i}" for i in range(1, 27)})

        cols = [str(c).strip() for c in df.columns.tolist()]
        mapped_cols = [rename_map.get(c, rename_map.get(c.lower(), c)) for c in cols]
        out = df.copy()
        out.columns = mapped_cols

        missing = [c for c in TRAIN_DATA_COLUMNS if c not in out.columns]
        if missing:
            raise ValueError(f"CSV header missing columns: {missing}")

        return out[TRAIN_DATA_COLUMNS]

    def fit(self, df: pd.DataFrame) -> None:
        """拟合所有稠密/稀疏处理器。

        示例:
            输入:
                train_df
            输出:
                无返回值，内部处理器参数被拟合
            原因解释:
                将参数学习与 transform 解耦，符合训练/验证分离流程。
        """
        for col in CONTINUOUS_COLUMNS:
            self.dense_processors[col].fit(df[col].fillna(self.fill_dense))
        for col in CATEGORICAL_COLUMNS:
            self.sparse_processors[col].fit(df[col].fillna(self.fill_sparse))
        self._is_fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """对 DataFrame 执行统一特征变换。

        示例:
            输入:
                val_df
            输出:
                处理后的 val_df（数值化特征）
            原因解释:
                对新数据复用已拟合参数，保证训练与推理一致。
        """
        df = df.copy()
        df[LABEL_COLUMN] = df[LABEL_COLUMN].fillna(0.0).astype(np.float32)

        for col in CONTINUOUS_COLUMNS:
            df[col] = self.dense_processors[col].transform(df[col].fillna(self.fill_dense))

        for col in CATEGORICAL_COLUMNS:
            df[col] = self.sparse_processors[col].transform(df[col].fillna(self.fill_sparse))

        return df

    def load_and_process(self, data_path: str, chunksize: Optional[int] = None) -> pd.DataFrame:
        """端到端执行：txt/csv 读取 + 拟合 + 转换。

        示例:
            输入:
                "./data/Criteo/train.txt"
            输出:
                处理后的完整 DataFrame
            原因解释:
                一步完成 txt->csv->特征处理，降低调用方复杂度。
        """
        csv_path = ensure_csv(data_path)
        print(f"Loading data from csv: {csv_path}")

        # chunksize=None: 全量读取并在全训练集上拟合，统计更稳定
        if chunksize is None:
            print("[Processor] chunksize=None，使用全量拟合模式（先全量 fit，再 transform）。")
            data = pd.read_csv(csv_path)
            data = self._normalize_columns(data)
            if not self._is_fitted:
                self.fit(data)
            return self.transform(data)

        if not isinstance(chunksize, int) or chunksize <= 0:
            raise ValueError("chunksize must be a positive integer or None")

        print(f"[Processor] chunksize={chunksize}，使用 chunk 模式。")
        print("Progress: [1~2/4] Reading csv file in chunks & Fill NaNs...")

        chunk_list = []
        df_iter = pd.read_csv(csv_path, chunksize=chunksize)

        fit_done = self._is_fitted
        for chunk in tqdm.tqdm(df_iter, desc="Reading & Processing Chunks"):
            chunk = self._normalize_columns(chunk)
            if not fit_done:
                self.fit(chunk)
                fit_done = True
            chunk_list.append(self.transform(chunk))

        print("Concatenating chunks into a single DataFrame...")
        data = pd.concat(chunk_list, ignore_index=True)
        del chunk_list
        return data
