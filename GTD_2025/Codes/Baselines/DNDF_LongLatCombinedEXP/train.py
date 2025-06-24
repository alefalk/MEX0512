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

    parser.add_argument('-lr', type=float, default=0.001, help="sgd: 10, adam: 0.001")
    parser.add_argument('-gpuid', type=int, default=-1)
    parser.add_argument('-verbose', type=int, default=0)
    parser.add_argument('-jointly_training', action='store_true', default=False)
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-report_every', type=int, default=10)

    opt = parser.parse_args()
    return opt

# MODIFIED
def prepare_db(opt):
    print("Use %s dataset" % (opt.dataset))

    if opt.dataset == 'gtd100':
        train_dataset = dataset.UCIgtd('../../../data/top30groups/LongLatCombined/scaledtrain1/train100.csv', target_col='gname')
        eval_dataset = dataset.UCIgtd('../../../data/top30groups/LongLatCombined/scaledtest1/test100.csv', target_col='gname')
        return {'train': train_dataset, 'eval': eval_dataset}

    elif opt.dataset == 'gtd200':
        train_dataset = dataset.UCIgtd('../../../data/top30groups/LongLatCombined/scaledtrain1/train200.csv', target_col='gname')
        eval_dataset = dataset.UCIgtd('../../../data/top30groups/LongLatCombined/scaledtest1/test200.csv', target_col='gname')
        return {'train': train_dataset, 'eval': eval_dataset}

    elif opt.dataset == 'gtd300':
        train_dataset = dataset.UCIgtd('../../../data/top30groups/LongLatCombined/scaledtrain1/train300.csv', target_col='gname')
        eval_dataset = dataset.UCIgtd('../../../data/top30groups/LongLatCombined/scaledtest1/test300.csv', target_col='gname')
        return {'train': train_dataset, 'eval': eval_dataset}

    elif opt.dataset == 'gtd478':
        train_dataset = dataset.UCIgtd('../../../data/top30groups/LongLatCombined/scaledtrain1/train478.csv', target_col='gname')
        eval_dataset = dataset.UCIgtd('../../../data/top30groups/LongLatCombined/scaledtest1/test478.csv', target_col='gname')
        return {'train': train_dataset, 'eval': eval_dataset}
    else:
        raise NotImplementedError


def prepare_model(opt):
    if opt.dataset == 'gtd100':
        feat_layer = ndf.GTDFeatureLayer100(opt.feat_dropout)
    elif opt.dataset == 'gtd200':
        feat_layer = ndf.GTDFeatureLayer200(opt.feat_dropout)
    elif opt.dataset == 'gtd300':
        feat_layer = ndf.GTDFeatureLayer300(opt.feat_dropout)
    elif opt.dataset == 'gtd478':
        feat_layer = ndf.GTDFeatureLayer478(opt.feat_dropout)
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


def train(model, optim, db, opt):
    torch.cuda.empty_cache()
    open(f"results/result_{opt.dataset}", "w")
    max = 0
    max_epoch = 0
    best_preds = []
    best_targets = []
    best_labels = []
    epoch_logs = []
    no_improvement_count = 0
    patience = 100
    for epoch in tqdm(range(1, opt.epochs + 1), desc="Training Epochs"):
        start_time = time.time()
        # Update \Theta
        model.train()
        train_loader = torch.utils.data.DataLoader(db['train'], batch_size=opt.batch_size, shuffle=True)
        for batch_idx, (data, target) in enumerate(train_loader):
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optim.zero_grad()
            output = model(data)
            loss = F.nll_loss(torch.log(output), target)
            loss.backward()
            # torch.nn.utils.clip_grad_norm([ p for p in model.parameters() if p.requires_grad],
            #                              max_norm=5)
            optim.step()
            if batch_idx % opt.report_every == 0 and opt.verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

        # Eval
        model.eval()
        test_loss = 0
        correct = 0
        all_preds = []
        all_targets = []
        all_probs = []

        test_loader = torch.utils.data.DataLoader(db['eval'], batch_size=opt.batch_size, shuffle=False)
        with torch.no_grad():
            for data, target in test_loader:
                if opt.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                output = model(data)
                probs = output.detach().cpu().numpy()
                all_probs.extend(probs)
                test_loss += F.nll_loss(torch.log(output), target, reduction='sum').item()

                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

                all_preds.extend(pred.cpu().numpy().flatten().tolist())
                all_targets.extend(target.cpu().numpy().flatten().tolist())

        # Binarize true labels for ROC AUC
        all_targets_bin = label_binarize(all_targets, classes=list(range(opt.n_class)))

        # Compute weighted multiclass ROC AUC
        roc_auc_weighted = roc_auc_score(all_targets_bin, all_probs, average='weighted', multi_class='ovr')
        roc_auc_micro = roc_auc_score(all_targets_bin, all_probs, average='micro', multi_class='ovr')
        roc_auc_macro = roc_auc_score(all_targets_bin, all_probs, average='macro', multi_class='ovr')

        test_loss /= len(test_loader.dataset)

        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        microPrecision = precision_score(all_targets, all_preds, average='micro', zero_division=0)
        microRecall = recall_score(all_targets, all_preds, average='micro', zero_division=0)
        microF1 = f1_score(all_targets, all_preds, average='micro', zero_division=0)
        macroPrecision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        macroRecall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        macroF1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        accuracy = correct / len(test_loader.dataset)

        text = (
            f'\nTest set: Average loss: {test_loss:.4f}, '
            f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.4f})\n'
        )
        #print(text)

        if max < (correct / len(test_loader.dataset)):
            max = (correct / len(test_loader.dataset))
            max_epoch = epoch

            best_precision = precision
            best_recall = recall
            best_f1 = f1
            best_rocauc = roc_auc_weighted

            best_precision_micro = microPrecision
            best_recall_micro = microRecall
            best_f1_micro = microF1
            best_rocauc_micro = roc_auc_micro

            best_precision_macro = macroPrecision
            best_recall_macro = macroRecall
            best_f1_macro = macroF1
            best_rocauc_macro = roc_auc_macro

            best_preds = all_preds.copy()
            best_targets = all_targets.copy()
            best_labels = db['eval'].labels  # list of label names from test set

            torch.save(model.state_dict(), f"results/best_model_{opt.dataset}.pt")
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        with open(f"results/result_{opt.dataset}", "a") as f:
            f.write(f'Epoch {epoch}: {text}') 

        epoch_time = time.time() - start_time
        epoch_logs.append(epoch_time)                        

    with open(f"results/result_{opt.dataset}", "a") as f:
        f.write(f'\nBest Accuracy: {max:.6f} at epoch {max_epoch}\n')
        f.write(f'Weighted Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1 Score: {best_f1:.4f}, ROCAUC: {best_rocauc:.4f}\n')
        f.write(f'macro Precision: {best_precision_macro:.4f}, Recall: {best_recall_macro:.4f}, F1 Score: {best_f1_macro:.4f}, ROCAUC: {best_rocauc_macro:.4f}\n')
        f.write(f'micro Precision: {best_precision_micro:.4f}, Recall: {best_recall_micro:.4f}, F1 Score: {best_f1_micro:.4f}, ROCAUC: {best_rocauc_micro:.4f}\n')

    dir_path = f"results/epoch_times_{opt.dataset}"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    epoch_file_path = os.path.join(dir_path, "log.txt")

    with open(epoch_file_path, "w") as file:
        file.write('\n'.join(str(x) for x in epoch_logs))

    model.load_state_dict(torch.load(f"results/best_model_{opt.dataset}.pt"))
    best_model = model

    return best_model, best_preds, best_targets, best_labels, epoch_logs


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
    best_model, best_preds, best_targets, best_labels, epoch_logs = train(model, optim, db, opt)
    return best_model, best_preds, best_targets, best_labels, epoch_logs


if __name__ == '__main__':
    main()
