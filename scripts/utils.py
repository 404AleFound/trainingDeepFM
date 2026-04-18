import os
import random
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
    
seed_everything(42)
print("set global seed: 42")


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