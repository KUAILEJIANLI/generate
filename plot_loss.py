import matplotlib.pyplot as plt
import re
import argparse
import os

def parse_log(log_path):
    epochs, total_loss, int_loss, grad_loss = [], [], [], []
    # 正则匹配日志中的 Summary 行
    summary_pattern = re.compile(r"📊 Epoch (\d+) summary: Total=([\d.]+), Int=([\d.]+), Grad=([\d.]+)")
    
    if not os.path.exists(log_path):
        print(f"❌ Log file not found: {log_path}")
        return
    
    with open(log_path, 'r') as f:
        for line in f:
            match = summary_pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                total_loss.append(float(match.group(2)))
                int_loss.append(float(match.group(3)))
                grad_loss.append(float(match.group(4)))
    return epochs, total_loss, int_loss, grad_loss

def plot(log_path):
    data = parse_log(log_path)
    if not data or len(data[0]) == 0: return
    epochs, total, intensity, gradient = data

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, total, label='Total Loss', marker='o')
    plt.plot(epochs, intensity, label='Intensity Loss (IR)', linestyle='--')
    plt.plot(epochs, gradient, label='Gradient Loss (Vis)', linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('SALD Stage 2: Training Loss Trends')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_name = log_path.replace('.log', '.png')
    plt.savefig(save_name)
    print(f"📈 Plot saved as: {save_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True, help="Path to your .log file")
    args = parser.parse_args()
    plot(args.log)