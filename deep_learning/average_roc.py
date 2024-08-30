# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# import os

# def plot_average_roc_curve(result_dir, num_folds):
#     all_labels = []
#     all_preds = []
    
#     for fold_num in range(num_folds):
#         fold_labels = np.load(os.path.join(result_dir, f'pdc_fold_{fold_num}_labels.npy'))
#         fold_preds = np.load(os.path.join(result_dir, f'pdc_fold_{fold_num}_preds.npy'))
#         all_labels.extend(fold_labels)
#         all_preds.extend(fold_preds)
    
#     all_labels = np.array(all_labels)
#     all_preds = np.array(all_preds)
    
#     fpr, tpr, _ = roc_curve(all_labels, all_preds)
#     roc_auc = auc(fpr, tpr)
    
#     plt.figure()
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label='Average ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic - Average')
#     plt.legend(loc="lower right")
#     plt.savefig(os.path.join(result_dir, 'average_pdc_roc_curve.png'))
#     plt.close()

# if __name__ == "__main__":
#     result_dir = './roc_curve'  # Adjust this path to your result directory
#     num_folds = 5
#     plot_average_roc_curve(result_dir, num_folds)
    
    
    
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

def plot_average_roc_curve(models,names, result_dir, num_folds):
    plt.figure()

    for model, name in zip(models, names):
        all_labels = []
        all_preds = []
        
        for fold_num in range(num_folds):
            fold_labels = np.load(os.path.join(result_dir, f'{model}_fold_{fold_num}_labels.npy'))
            fold_preds = np.load(os.path.join(result_dir, f'{model}_fold_{fold_num}_preds.npy'))
            all_labels.extend(fold_labels)
            all_preds.extend(fold_preds)
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{name} ROC curve (area = %0.2f)' % roc_auc)
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve and AUC for each model')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(result_dir, 'jmir_ROC.png'))
    plt.close()

if __name__ == "__main__":
    result_dir = './roc_curve'  # Adjust this path to your result directory
    num_folds = 5
    models = ['jmir_sethoscope', 'jmir_iphone', 'jmir_steth_iphone','jmir_dat']  # List of model names
    name = ['AST(Stethoscope)','AST(Smartphone)','AST(Combined data)','AST+DAT(Combined data)']
    plot_average_roc_curve(models, name,result_dir, num_folds)
