"""Utility functions for visualizing model performance metrics"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_pr_curve(precision, recall, auprc):
    """Plot Precision-Recall curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.axhline(y=sum(y_test)/len(y_test), color='r', linestyle='--', label='No Skill')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUPRC = {auprc:.3f})')
    plt.grid(True)
    plt.legend()
    return plt.gcf()

def plot_roc_curve(fpr, tpr, roc_auc):
    """Plot ROC curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='r', linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    return plt.gcf()

def plot_confusion_matrix(tn, fp, fn, tp):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(8, 6))
    cm = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    return plt.gcf()

def plot_metric_curves(metrics_dict):
    """Plot all relevant metric curves"""
    # Create PR curve
    pr_fig = plot_pr_curve(
        metrics_dict['curves']['pr']['precision'],
        metrics_dict['curves']['pr']['recall'],
        metrics_dict['auprc']
    )
    
    # Create ROC curve
    roc_fig = plot_roc_curve(
        metrics_dict['curves']['roc']['fpr'],
        metrics_dict['curves']['roc']['tpr'],
        metrics_dict['roc_auc']
    )
    
    # Create confusion matrix
    cm_fig = plot_confusion_matrix(
        metrics_dict['true_negatives'],
        metrics_dict['false_positives'],
        metrics_dict['false_negatives'],
        metrics_dict['true_positives']
    )
    
    return pr_fig, roc_fig, cm_fig