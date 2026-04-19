import os
import random
import csv
import hashlib
from collections import deque
import numpy as np
import torch    

def seed_everything(seed: int = 42):
    """
    Fix all random seeds to ensure complete reproducibility of code and experiments.
    Includes Python native, Numpy, PyTorch(CPU/GPU) and CuDNN backend.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 针对使用 CuDNN 的场景，关闭底层算法的自动寻优，固定随机算子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def sample_text_file(input_path, output_path, num_samples, seed=42):
    """Randomly sample lines from a text file and write them out."""
    import random
    from tqdm import tqdm

    # Reservoir sampling keeps memory usage bounded even for large files.
    reservoir = []
    random.seed(seed)

    with open(input_path, "r", encoding="utf-8") as f_in:
        for idx, line in enumerate(tqdm(f_in, desc="Sampling", unit=" lines")):
            if idx < num_samples:
                reservoir.append(line)
            else:
                j = random.randint(0, idx)
                if j < num_samples:
                    reservoir[j] = line

    if len(reservoir) < num_samples:
        raise ValueError(
            f"Not enough samples: requested {num_samples}, got {len(reservoir)}"
        )

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.writelines(reservoir)


def split_csv_by_ratio(
    input_path,
    output_train_path,
    output_val_path,
    val_ratio=0.1,
    seed=42,
    has_header=True,
):
    """Split a CSV file into train/val subsets by ratio."""
    import csv

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    random.seed(seed)

    with open(input_path, "r", encoding="utf-8", newline="") as f_in, \
        open(output_train_path, "w", encoding="utf-8", newline="") as f_train, \
        open(output_val_path, "w", encoding="utf-8", newline="") as f_val:
        reader = csv.reader(f_in)
        train_writer = csv.writer(f_train)
        val_writer = csv.writer(f_val)

        if has_header:
            header = next(reader, None)
            if header:
                train_writer.writerow(header)
                val_writer.writerow(header)

        for row in reader:
            if random.random() < val_ratio:
                val_writer.writerow(row)
            else:
                train_writer.writerow(row)


def check_csv_overlap(train_csv_path, val_csv_path, has_header=True):
    """Check whether two CSV files overlap and return overlap statistics.

    Args:
        train_csv_path: Path to train CSV file.
        val_csv_path: Path to validation CSV file.
        has_header: Whether both CSV files contain a header row.

    Returns:
        A dictionary containing overlap flag, overlap count and overlap rates.
    """

    def _row_hash(row):
        # Use a stable delimiter to avoid collisions from naive string join.
        row_str = "\x1f".join([str(x).strip() for x in row])
        return hashlib.md5(row_str.encode("utf-8")).hexdigest()

    train_hashes = set()
    val_hashes = set()

    with open(train_csv_path, "r", encoding="utf-8", newline="") as f_train:
        train_reader = csv.reader(f_train)
        if has_header:
            next(train_reader, None)
        for row in train_reader:
            if not row:
                continue
            train_hashes.add(_row_hash(row))

    with open(val_csv_path, "r", encoding="utf-8", newline="") as f_val:
        val_reader = csv.reader(f_val)
        if has_header:
            next(val_reader, None)
        for row in val_reader:
            if not row:
                continue
            val_hashes.add(_row_hash(row))

    overlap_count = len(train_hashes & val_hashes)
    train_count = len(train_hashes)
    val_count = len(val_hashes)
    union_count = len(train_hashes | val_hashes)

    train_overlap_rate = overlap_count / train_count if train_count > 0 else 0.0
    val_overlap_rate = overlap_count / val_count if val_count > 0 else 0.0
    jaccard_overlap_rate = overlap_count / union_count if union_count > 0 else 0.0

    return {
        "has_overlap": overlap_count > 0,
        "overlap_count": overlap_count,
        "train_unique_count": train_count,
        "val_unique_count": val_count,
        "train_overlap_rate": train_overlap_rate,
        "val_overlap_rate": val_overlap_rate,
        "jaccard_overlap_rate": jaccard_overlap_rate,
    }


def split_txt_by_time_order(
    input_path,
    output_train_path,
    output_test_path,
    train_size=1_000_000,
    test_size=100_000,
    take_from="tail",
    has_header=False,
):
    """Split a huge txt file by time order into train/test subsets.

    Time concept used here:
        - Each line order in file is treated as timeline order.
        - train is always earlier part, test is always later part.

    Args:
        input_path: Source txt path.
        output_train_path: Output train txt path.
        output_test_path: Output test txt path.
        train_size: Number of lines for train split.
        test_size: Number of lines for test split.
        take_from:
            - "head": take earliest (front) train_size+test_size lines.
            - "tail": take latest (end) train_size+test_size lines.
        has_header: Whether the first line is header and should be copied.

    Returns:
        A dict with split statistics.
    """
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    if take_from not in {"head", "tail"}:
        raise ValueError("take_from must be either 'head' or 'tail'")

    total_needed = train_size + test_size
    header = None
    total_lines = 0

    if take_from == "head":
        selected = []
        with open(input_path, "r", encoding="utf-8") as f_in:
            if has_header:
                header = next(f_in, None)
            for line in f_in:
                if not line:
                    continue
                selected.append(line)
                total_lines += 1
                if len(selected) >= total_needed:
                    break
    else:
        window = deque(maxlen=total_needed)
        with open(input_path, "r", encoding="utf-8") as f_in:
            if has_header:
                header = next(f_in, None)
            for line in f_in:
                if not line:
                    continue
                window.append(line)
                total_lines += 1
        selected = list(window)

    if len(selected) < total_needed:
        raise ValueError(
            f"Not enough lines for time split: required {total_needed}, got {len(selected)}"
        )

    train_lines = selected[:train_size]
    test_lines = selected[train_size:train_size + test_size]

    with open(output_train_path, "w", encoding="utf-8") as f_train:
        if has_header and header is not None:
            f_train.write(header)
        f_train.writelines(train_lines)

    with open(output_test_path, "w", encoding="utf-8") as f_test:
        if has_header and header is not None:
            f_test.write(header)
        f_test.writelines(test_lines)

    return {
        "mode": "time_order_split",
        "take_from": take_from,
        "train_size": train_size,
        "test_size": test_size,
        "total_needed": total_needed,
        "source_lines_scanned": total_lines,
        "output_train_path": output_train_path,
        "output_test_path": output_test_path,
    }
        
    
if __name__ == "__main__":
    # Example usage:
    # seed_everything(123)
    # sample_text_file("input.txt", "sampled.txt", num_samples=100, seed=123)
    # split_csv_by_ratio("data.csv", "train.csv", "val.csv", val_ratio=0.2, seed=123)
    split_stats = split_txt_by_time_order(
        input_path="data/Criteo/train.txt",
        output_train_path="data/Criteo_sample/train_1m.txt",
        output_test_path="data/Criteo_sample/test_100k.txt",
        train_size=1_000_000,
        test_size=100_000,
        take_from="tail",
        has_header=False,
    )
    print(split_stats)
    pass

def deleteInvalidLogs(log_dir, required_tail_prefix="Training complete.  Best Val AUC:"):
    """Delete .log files whose ending text is not the required training-complete marker.

    Args:
        log_dir: Directory containing log files.
        required_tail_prefix: Required prefix in the last non-empty line of a valid log file.

    Returns:
        A list of deleted log file paths.
    """
    deleted_files = []

    for filename in os.listdir(log_dir):
        if not filename.lower().endswith(".log"):
            continue

        file_path = os.path.join(log_dir, filename)
        if not os.path.isfile(file_path):
            continue

        # Read only the tail chunk for efficiency on large log files.
        try:
            with open(file_path, "rb") as f:
                try:
                    f.seek(-4096, os.SEEK_END)
                except OSError:
                    f.seek(0)
                tail_text = f.read().decode("utf-8", errors="ignore")
        except OSError:
            tail_text = ""

        lines = [line.strip() for line in tail_text.splitlines() if line.strip()]
        last_line = lines[-1] if lines else ""

        if not last_line.startswith(required_tail_prefix):
            os.remove(file_path)
            deleted_files.append(file_path)

    return deleted_files

if __name__ == "__main__":
    # Example usage:
    # deleted = deleteInvalidLogs("./logger")
    # print(f"Deleted {len(deleted)} invalid log files:")
    # for path in deleted:
    #     print(f" - {path}")
    
    