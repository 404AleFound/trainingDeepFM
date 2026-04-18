import os
import re
import glob
import matplotlib.pyplot as plt

def parse_log_file(log_path):
    """
    Parse the specified log file to extract step-level loss and epoch-level metrics.
    """
    step_losses = []
    epoch_metrics = {
        'epoch': [],
        'train_loss': [],
        'val_auc': [],
        'val_logloss': [],
        'val_acc': [],
        'val_pre': [],
        'val_rec': [],
        'val_f1': []
    }

    # 正则表达式匹配 step level loss (例如: Epoch [1] Step [0/71] Train Loss: 0.6382)
    step_pattern = re.compile(r'Epoch \[(\d+)\] Step \[\d+/\d+\] Train Loss: ([0-9.]+)')
    
    # 正则表达式匹配 epoch level metrics 
    # (例如: Epoch [ 1/ 5] | Train Loss: 0.5213 | Val AUC: 0.7321 | LogLoss: 0.4901 | ACC: 0.8123 | Pre: 0.0000 | Rec: 0.0000 | F1: 0.0000 | LR: 1.00e-03)
    epoch_pattern = re.compile(
        r'Epoch \[\s*(\d+)/\s*\d+\]\s*\|\s*'
        r'Train Loss:\s*([0-9.]+)\s*\|\s*'
        r'Val AUC:\s*([0-9.]+)\s*\|\s*'
        r'LogLoss:\s*([0-9.]+)\s*\|\s*'
        r'ACC:\s*([0-9.]+)\s*\|\s*'
        r'Pre:\s*([0-9.]+)\s*\|\s*'
        r'Rec:\s*([0-9.]+)\s*\|\s*'
        r'F1:\s*([0-9.]+)'
    )

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Check step pattern
            step_match = step_pattern.search(line)
            if step_match:
                step_losses.append(float(step_match.group(2)))
                continue
            
            # Check epoch pattern
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                epoch_metrics['epoch'].append(int(epoch_match.group(1)))
                epoch_metrics['train_loss'].append(float(epoch_match.group(2)))
                epoch_metrics['val_auc'].append(float(epoch_match.group(3)))
                epoch_metrics['val_logloss'].append(float(epoch_match.group(4)))
                epoch_metrics['val_acc'].append(float(epoch_match.group(5)))
                epoch_metrics['val_pre'].append(float(epoch_match.group(6)))
                epoch_metrics['val_rec'].append(float(epoch_match.group(7)))
                epoch_metrics['val_f1'].append(float(epoch_match.group(8)))
    
    return step_losses, epoch_metrics

def plot_metrics(log_path, save_dir='./logger/plots'):
    """
    Plot metrics using matplotlib and save as image files.
    """
    os.makedirs(save_dir, exist_ok=True)
    step_losses, epoch_metrics = parse_log_file(log_path)
    log_name = os.path.basename(log_path).replace('.log', '')

    if not step_losses and not epoch_metrics['epoch']:
        print(f"No metrics found in {log_path}.")
        return

    # 1. 绘制 Step 级别的 Train Loss
    if step_losses:
        plt.figure(figsize=(10, 5))
        plt.plot(step_losses, label='Train Loss (Step)', color='cornflowerblue', alpha=0.8)
        plt.xlabel('Global Steps')
        plt.ylabel('Loss (BCE)')
        plt.title('Step-level Training Loss')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        step_loss_file = os.path.join(save_dir, f'{log_name}_step_loss.png')
        plt.savefig(step_loss_file, dpi=300)
        plt.close()
        print(f"Saved: {step_loss_file}")

    # 2. 绘制 Epoch 级别的各种指标
    if epoch_metrics['epoch']:
        epochs = epoch_metrics['epoch']
        
        # 绘图 A：训练 Loss 与 验证 LogLoss 对比
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, epoch_metrics['train_loss'], marker='o', label='Train Loss', color='tomato', linewidth=2)
        plt.plot(epochs, epoch_metrics['val_logloss'], marker='s', label='Val LogLoss', color='teal', linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Epoch-level: Train Loss vs Val LogLoss')
        plt.xticks(epochs)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        epoch_loss_file = os.path.join(save_dir, f'{log_name}_epoch_loss.png')
        plt.savefig(epoch_loss_file, dpi=300)
        plt.close()
        print(f"Saved: {epoch_loss_file}")

        # 绘图 B：综合评估指标评分 (AUC, ACC, F1, Pre, Rec)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, epoch_metrics['val_auc'], marker='o', label='Val AUC', color='purple', linewidth=2)
        plt.plot(epochs, epoch_metrics['val_acc'], marker='s', label='Val ACC', color='green', linewidth=2)
        plt.plot(epochs, epoch_metrics['val_f1'], marker='^', label='Val F1', color='orange', linewidth=2)
        plt.plot(epochs, epoch_metrics['val_pre'], marker='v', label='Val Precision', color='brown', linestyle='-.')
        plt.plot(epochs, epoch_metrics['val_rec'], marker='p', label='Val Recall', color='cyan', linestyle='-.')
        
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.title('Epoch-level Validation Ranking & Classification Metrics')
        plt.xticks(epochs)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 将图例放在图外
        plt.tight_layout()
        epoch_metrics_file = os.path.join(save_dir, f'{log_name}_epoch_metrics.png')
        plt.savefig(epoch_metrics_file, dpi=300)
        plt.close()
        print(f"Saved: {epoch_metrics_file}")

def main():
    # 自动去找 logger 文件夹下面的最新 .log 文件
    log_files = glob.glob('./logger/*.log')
    if not log_files:
        print("No log files found in ./logger/ directory.")
        return
    
    # 找最近创建/修改的日志文件
    latest_log = max(log_files, key=os.path.getctime)
    print(f"Parsing and plotting for latest log: {latest_log}")
    plot_metrics(latest_log)

if __name__ == '__main__':
    main()