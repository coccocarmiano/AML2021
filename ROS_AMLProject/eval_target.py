
from tqdm import tqdm
import torch
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import random

#### Implement the evaluation on the target for the known/unknw separation

def evaluation(args, feature_extractor, rot_cls, obj_cls, get_rotation_classifiers, target_loader_eval, device):
    softmax = torch.nn.Softmax(dim=1)
    feature_extractor.eval()
    obj_cls.eval()

    if args.multihead:
        for head in rot_cls:
            head.eval()
    else:
        rot_cls.eval()
    
    ground_truths    = []
    normality_scores = []

    with torch.no_grad():
        for data, data_label, data_rot, data_rot_label in tqdm(target_loader_eval):
            data,     data_label     =  data.to(device),     data_label.to(device)
            data_rot = data_rot.to(device)

            data_label[data_label > 44] = 45

            feature_extractor_output     = feature_extractor(data)
            feature_extractor_output_rot = feature_extractor(data_rot)
            output_rot_output_cat        = torch.cat((feature_extractor_output, feature_extractor_output_rot), dim=1)

            l_score   = obj_cls(feature_extractor_output)
            l_preds   = torch.argmax(l_score, dim=1)

            ### Using inferred lables for rotation prediction ###
            rotation_classifiers = get_rotation_classifiers(l_preds)
            it = range(len(rotation_classifiers))
            r_score   = torch.vstack( [ rotation_classifiers[idx](output_rot_output_cat[idx]) for idx in it ])
            n_scores = softmax(r_score)
            n_score, _ = torch.max(n_scores, dim=1)
            
            ground_truths.append(data_label.item())
            normality_scores.append(n_score.item())
            

    # Convert GTs and scores computed ( on inferred labels) into tensors
    ground_truths = torch.tensor(ground_truths).to(device)
    normality_scores = torch.tensor(normality_scores).to(device)

    # Conert to Binary Task : 1 is known, 0 in unknown
    mask_known = ground_truths < 45
    mask_unknw = ground_truths > 44
    ground_truths[mask_known] = 1
    ground_truths[mask_unknw] = 0

    ## Display ROC AUC Value
    auc = roc_auc_score(ground_truths.cpu(), normality_scores.cpu())
    print(f"Computed ROC AUC: {auc:.4f}")
    

    mask_known = normality_scores >= args.threshold
    mask_unknw = normality_scores <  args.threshold

    normality_scores[mask_known] = 1
    normality_scores[mask_unknw] = 0
    print(f"Marked Known: {normality_scores.sum().item()} Actually Known: {ground_truths.sum().item()}")

    ## We now must save two datasets
    ## New Source Dataset, with Source + Unknown Samples
    ## New Target Dataset, with only Known Samples

    file_names = target_loader_eval.dataset.names
    labels = target_loader_eval.dataset.labels

    if not os.path.isdir('new_txt_list'):
        os.mkdir('new_txt_list')

    # rand = args.rand ok so rand is not in args
    # is in self which is not accessible so for now let's keep it random
    rand = random.randint(0, 1e5)

    target_unknw = open(f'new_txt_list/{args.source}_known_{str(rand)}.txt', 'w')
    target_known = open(f'new_txt_list/{args.target}_known_{str(rand)}.txt', 'w')

    pairs = zip(target_loader_eval.dataset.names, target_loader_eval.dataset.labels)
    # Ok so naming here is a little confusing so I'm leaving a note
    # We must add UNKNOWN samples to the SOURCE
    # UNKNWON samples are labeled as a 0
    # SOURCE_KNOWN is SOURCE + UNKNOWN (???)
    # Refer to `Project_OpenSet.pdf`
    known = normality_scores >= args.threshold
    for it, (name, label) in enumerate(pairs):
        if known[it] > 0:
            target_known.write(f"{name} {str(label)}\n")
        else:
            target_unknw.write(f"{name} {str(label)}\n")
    
    target_known.close()
    target_unknw.close()

    return rand






