import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_pr_curves(csv_files, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    for csv_file, label in zip(csv_files, labels):
        df = pd.read_csv(csv_file)
        plt.plot(df['recall'], df['precision'], label=label)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=300)
    plt.close()


def plot_roc_curves(csv_files, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    for csv_file, label in zip(csv_files, labels):
        df = pd.read_csv(csv_file)
        plt.plot(df['FPR'], df['TPR'], label=label)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot PR and ROC curves from macro-averaged CSV files")
    parser.add_argument('--pr_csvs', type=str, nargs='+', required=True, help='PR macro CSV files (one per model)')
    parser.add_argument('--roc_csvs', type=str, nargs='+', required=True, help='ROC macro CSV files (one per model)')
    parser.add_argument('--labels', type=str, nargs='+', required=True, help='Model labels for legend')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for plots')

    args = parser.parse_args()
    plot_pr_curves(args.pr_csvs, args.labels, args.output_dir)
    plot_roc_curves(args.roc_csvs, args.labels, args.output_dir)
