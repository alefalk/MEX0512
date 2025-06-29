import argparse
import logging

import torch
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import os
import shutil
import dataset
import ndf
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import time

def parse_arg():
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(asctime)s]: %(levelname)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-dataset', choices=['gtd100', 'gtd200', 'gtd300', 'gtd478'], default='gtd100')
    parser.add_argument('-batch_size', type=int, default=128)

    parser.add_argument('-feat_dropout', type=float, default=0.3)

    parser.add_argument('-n_tree', type=int, default=5)
    parser.add_argument('-tree_depth', type=int, default=3)
    parser.add_argument('-n_class', type=int, default=10)
    parser.add_argument('-tree_feature_rate', type=float, default=0.5)
    parser.add_argument('-hidden_dim', type=int, default=1024)

    parser.add_argument('-lr', type=float, default=0.0001, help="sgd: 10, adam: 0.001")
    parser.add_argument('-gpuid', type=int, default=-1)
    parser.add_argument('-verbose', type=int, default=0)
    parser.add_argument('-jointly_training', action='store_true', default=False)
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-report_every', type=int, default=10)
    parser.add_argument('-searching', type=int, default=0)
    parser.add_argument('-results_dir', type=str, default='results', help='Path to save results')


    opt = parser.parse_args()
    return opt

# MODIFIED
def prepare_db(opt):
    print("Use %s dataset" % (opt.dataset))

    if opt.dataset == 'gtd100':
        train_data_path = '../../../data/top30groups/LongLatCombined/scaledtrain1/train100.csv'
        test_data_path = '../../../data/top30groups/LongLatCombined/scaledtest1/test100.csv'
    elif opt.dataset == 'gtd200':
        train_data_path = '../../../data/top30groups/LongLatCombined/scaledtrain1/train200.csv'
        test_data_path = '../../../data/top30groups/LongLatCombined/scaledtest1/test200.csv'
    elif opt.dataset == 'gtd300':
        train_data_path = '../../../data/top30groups/LongLatCombined/scaledtrain1/train300.csv'
        test_data_path = '../../../data/top30groups/LongLatCombined/scaledtest1/test300.csv'
    elif opt.dataset == 'gtd478':
        train_data_path = '../../../data/top30groups/LongLatCombined/scaledtrain1/train478.csv'
        test_data_path = '../../../data/top30groups/LongLatCombined/scaledtest1/test478.csv'
    else:
        raise NotImplementedError

    # Add eval set
    full_train_dataset = dataset.UCIgtd(train_data_path, target_col='gname')
    test_dataset = dataset.UCIgtd(test_data_path, target_col='gname')

    indices = list(range(len(full_train_dataset)))
    train_idx, eval_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=full_train_dataset.targets
    )
    train_dataset = Subset(full_train_dataset, train_idx)
    eval_dataset = Subset(full_train_dataset, eval_idx)

    return {'train': train_dataset, 'eval': eval_dataset, 'test': test_dataset}

def prepare_model(opt):
    if opt.dataset == 'gtd100':
        feat_layer = ndf.GTDFeatureLayer100(opt.hidden_dim, opt.feat_dropout)
    elif opt.dataset == 'gtd200':
        feat_layer = ndf.GTDFeatureLayer200(opt.hidden_dim, opt.feat_dropout)
    elif opt.dataset == 'gtd300':
        feat_layer = ndf.GTDFeatureLayer300(opt.hidden_dim, opt.feat_dropout)
    elif opt.dataset == 'gtd478':
        feat_layer = ndf.GTDFeatureLayer478(opt.hidden_dim, opt.feat_dropout)
    else:
        raise NotImplementedError

    forest = ndf.Forest(n_tree=opt.n_tree, tree_depth=opt.tree_depth, n_in_feature=feat_layer.get_out_feature_size(),
                        tree_feature_rate=opt.tree_feature_rate, n_class=opt.n_class,
                        jointly_training=opt.jointly_training)
    model = ndf.NeuralDecisionForest(feat_layer, forest)
    #print(model.feature_layer)

    if opt.cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    return model


def prepare_optim(model, opt):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=opt.lr, weight_decay=1e-5)

# Only used when we have searching = 0
def evaluate_on_test(model, test_loader, opt, out_path):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for data, target in test_loader:
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            probs = output.detach().cpu().numpy()
            preds = output.argmax(dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(target.cpu().numpy())

    all_targets_bin = label_binarize(all_targets, classes=list(range(opt.n_class)))
    acc = np.mean(np.array(all_preds) == np.array(all_targets))

    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    roc_auc_weighted = roc_auc_score(all_targets_bin, all_probs, average='weighted', multi_class='ovr')
    roc_auc_micro = roc_auc_score(all_targets_bin, all_probs, average='micro', multi_class='ovr')
    roc_auc_macro = roc_auc_score(all_targets_bin, all_probs, average='macro', multi_class='ovr')

    microPrecision = precision_score(all_targets, all_preds, average='micro', zero_division=0)
    microRecall = recall_score(all_targets, all_preds, average='micro', zero_division=0)
    microF1 = f1_score(all_targets, all_preds, average='micro', zero_division=0)

    macroPrecision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    macroRecall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    macroF1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    with open(out_path, "a") as f:
        f.write("\n========== Final Test Evaluation ==========\n")
        f.write(f"Model Parameters:\n")
        f.write(f"  Dataset: {opt.dataset}\n")
        f.write(f"  Hidden Dim: {opt.hidden_dim}\n")
        f.write(f"  n_tree: {opt.n_tree}, tree_depth: {opt.tree_depth}, tree_feature_rate: {opt.tree_feature_rate}\n")
        f.write(f"  Batch size: {opt.batch_size}, Dropout: {opt.feat_dropout}, LR: {opt.lr}\n")
        f.write(f"\nBest Accuracy: {acc:.4f}\n")
        f.write(f"Weighted Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROCAUC: {roc_auc_weighted:.4f}\n")
        f.write(f"Macro Precision: {macroPrecision:.4f}, Recall: {macroRecall:.4f}, F1 Score: {macroF1:.4f}, ROCAUC: {roc_auc_macro:.4f}\n")
        f.write(f"Micro Precision: {microPrecision:.4f}, Recall: {microRecall:.4f}, F1 Score: {microF1:.4f}, ROCAUC: {roc_auc_micro:.4f}\n")

    return all_preds, all_targets

    #print(f"Test Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")


def train(model, optim, db, opt):
    torch.cuda.empty_cache()
    os.makedirs(opt.results_dir, exist_ok=True)
    result_path = os.path.join(opt.results_dir, f"result_{opt.dataset}.txt")
    best_model_path = os.path.join(opt.results_dir, f"best_model_{opt.dataset}.pt")

    with open(result_path, "w") as f:
        pass  

    max = 0
    best_preds, best_targets, best_labels = [], [], []
    epoch_logs = []
    no_improvement_count = 0
    patience = 300 if not opt.searching else 100

    print("Patience:", patience)

    for epoch in tqdm(range(1, opt.epochs + 1), desc="Training Epochs"):
        start_time = time.time()
        model.train()
        train_loader = torch.utils.data.DataLoader(db['train'], batch_size=opt.batch_size, shuffle=True)

        train_loss_total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optim.zero_grad()
            output = model(data)
            loss = F.nll_loss(torch.log(output), target)
            train_loss_total += loss.item()
            loss.backward()
            optim.step()
        avg_train_loss = train_loss_total / len(train_loader)

        # Eval
        model.eval()
        eval_loader = torch.utils.data.DataLoader(db['eval'], batch_size=opt.batch_size, shuffle=False)
        eval_loss, correct = 0, 0
        all_preds, all_targets, all_probs = [], [], []

        with torch.no_grad():
            for data, target in eval_loader:
                if opt.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                output = model(data)
                probs = output.detach().cpu().numpy()
                eval_loss += F.nll_loss(torch.log(output), target, reduction='sum').item()

                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()

                all_probs.extend(probs)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        eval_loss /= len(eval_loader.dataset)
        acc = correct / len(eval_loader.dataset)

        # log training progress for final evaluation
        if not opt.searching and epoch % 50 == 0:
            print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f}, Eval Loss: {eval_loss:.4f}, Eval Accuracy: {acc:.4f}")

        if acc > max:
            max = acc
            torch.save(model.state_dict(), best_model_path)
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        epoch_logs.append(time.time() - start_time)

    model.load_state_dict(torch.load(best_model_path))
    best_model = model

    if not opt.searching:
        print("Evaluating on test set with best model...")
        test_loader = torch.utils.data.DataLoader(db['test'], batch_size=opt.batch_size, shuffle=False)
        best_preds, best_targets = evaluate_on_test(best_model, test_loader, opt, result_path)

        # Decode labels
        label_names = db['test'].labels  
        decoded_targets = [label_names[i] for i in best_targets]
        decoded_preds = [label_names[i] for i in best_preds]

        return best_model, decoded_preds, decoded_targets, db['test'].labels, epoch_logs

    if opt.searching:
        print(f"\nBest Accuracy: {max:.6f}")
        with open(result_path, "a") as f:
            f.write(f"\nBest Accuracy: {max:.6f}\n")
            #f.write(str(best_model))

    print(f"Results written to: {opt.results_dir}")

    return "complete"


def main():
    opt = parse_arg()

    # GPU
    opt.cuda = opt.gpuid >= 0
    if opt.gpuid >= 0:
        torch.cuda.set_device(opt.gpuid)
    else:
        print("WARNING: RUN WITHOUT GPU")

    db = prepare_db(opt)
    model = prepare_model(opt)
    optim = prepare_optim(model, opt)
    if not opt.searching:
        best_model, best_preds, best_targets, best_labels, epoch_logs = train(model, optim, db, opt)
        return best_model, best_preds, best_targets, best_labels, epoch_logs
    else:
        complete = train(model, optim, db, opt)
        return complete


if __name__ == '__main__':
    main()
