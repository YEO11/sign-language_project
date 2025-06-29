# utils/plot_utils.py
import matplotlib.pyplot as plt

def save_metrics_plot(losses, accuracies, filename):
    plt.figure()
    plt.plot(losses, label='Loss')
    plt.plot(accuracies, label='Accuracy')
    plt.xlabel("Epoch")
    plt.title("Training Metrics")
    plt.legend()
    plt.savefig(filename)
    plt.close()
