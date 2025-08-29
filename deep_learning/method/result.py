import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import os
from sklearn.manifold import TSNE
import torch

def process_results(results_dict, save_dir, class_names):
    """
    Process and save confusion matrices and ROC curves for multiple models
    
    Args:
        results_dict: Dictionary containing results for each model
            format: {
                'model_name': {
                    'true_labels': array,
                    'pred_labels': array,
                    'pred_probs': array
                }
            }
        save_dir: Directory to save the results
        class_names: List of class names
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    for model_name, results in results_dict.items():
        # 결과가 비어있지 않은지 확인
        if (len(results['true_labels']) == 0 or 
            len(results['pred_labels']) == 0 or 
            len(results['pred_probs']) == 0):
            print(f"Skipping {model_name} due to empty results")
            continue
            
        print(f"Processing results for {model_name}")
        print(f"Number of samples: {len(results['true_labels'])}")
        
        # 혼동 행렬 생성 및 저장
        try:
            plot_confusion_matrix(
                y_true=results['true_labels'],
                y_pred=results['pred_labels'],
                classes=class_names,
                save_path=os.path.join(save_dir, f'confusion_matrix_{model_name}.png'),
                model_name=model_name
            )
        except Exception as e:
            print(f"Error creating confusion matrix for {model_name}: {str(e)}")
            
        # ROC 커브 생성 및 저장 (이진 분류의 경우)
        try:
            if results['pred_probs'].shape[1] == 2:  # 이진 분류인 경우
                plot_and_save_roc_curve(
                    y_true=results['true_labels'],
                    y_score=results['pred_probs'][:, 1],
                    save_path=os.path.join(save_dir, f'roc_curve_{model_name}.png')
                )
        except Exception as e:
            print(f"Error creating ROC curve for {model_name}: {str(e)}")

def plot_confusion_matrix(y_true, y_pred, classes, save_path, model_name):
    """
    Plot and save confusion matrix
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        print("Error: Empty input arrays")
        return
    
     # 혼동 행렬 계산
    cm = confusion_matrix(y_true, y_pred)
    
    # 정규화
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 폰트 크기 설정
    plt.rcParams.update({'font.size': 20})  # 기본 폰트 크기 증가


    # 그래프 생성
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'{model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # 저장 및 닫기
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    # tick 레이블 폰트 크기 설정
    

def plot_and_save_roc_curve(y_true, y_score, save_path):
    if len(y_true) == 0 or len(y_score) == 0:
        print("Error: Empty input arrays")
        return
    
    # ROC 커브 계산
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.rcParams.update({'font.size': 20})  # 기본 폰트 크기 증가
    plt.figure(figsize=(10, 8))
    
    # 그래프 생성
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # 저장 및 닫기
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()





def visualize_embeddings(model,save_dir,model_name, train_loader, args):
    """
    Extract and visualize embeddings using t-SNE
    Args:
        model: Neural network model
        train_loader: DataLoader containing training data
        args: Arguments containing save directory information
    """

    # Create subdirectories
    tsne_dir = os.path.join(save_dir, 'tsne')
    os.makedirs(tsne_dir, exist_ok=True)


    model.eval()
    embeddings = []
    domains = []
    labels = []
    
    with torch.no_grad():
        for idx, (images, target) in enumerate(train_loader):
            if idx % 10 == 0:  # Print progress every 10 batches
                print(f'Processing batch {idx}/{len(train_loader)}')
            
            images = images.cuda(non_blocking=True)
            features = model(images, args=args, training=False)
            embeddings.append(features.cpu().numpy())
            
            # Handle different target formats
            if isinstance(target, list):
                domains.extend(target[1].numpy())  # Device/domain information
                labels.extend(target[0].numpy())   # Class labels
            else:
                labels.extend(target.numpy())
                domains.extend([0] * len(target))  # Default domain if not provided

    embeddings = np.vstack(embeddings)
    domains = np.array(domains)
    labels = np.array(labels)
    
    # Perform t-SNE
    print('Performing t-SNE dimensionality reduction...')
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Plot by domain
    plt.figure(figsize=(20, 16))
    unique_domains = np.unique(domains)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_domains)))
    
    for domain, color in zip(unique_domains, colors):
        mask = domains == domain
        plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], 
                   c=[color], label=f'Device {domain}', alpha=0.6, s=10)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f't-SNE visualization - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(tsne_dir, f'tsne_device_embeddings_{model_name}.png'), dpi=300)
    plt.close()
    
    # Plot by class
    plt.figure(figsize=(20, 16))
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], 
                   c=[color], label=f'Class {label}', alpha=0.6, s=10)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f't-SNE visualization - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(tsne_dir, f'tsne_class_embeddings_{model_name}.png'), dpi=300)
    plt.close()

    print('Visualization completed.')