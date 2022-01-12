
import torch
import numpy as np
import os
from sklearn.metrics import roc_auc_score
import random

#### Implement the evaluation on the target for the known/unknw separation

def evaluation(args, feature_extractor, rot_cls, obj_cls, get_rotation_classifiers, target_loader_eval, device):

    feature_extractor.eval()
    rot_cls.eval()

    if args.multihead:
        for head in rot_cls:
            head.eval()
        else:
            rot_cls.eval()
    
    ground_truths, normality_scores = [], []

    with torch.no_grad():
        for it, (data, data_label, data_rot, data_rot_label) in enumerate(target_loader_eval):
            data,     data_label     =  data.to(device),     data_label.to(device)
            data_rot, data_rot_label =  data_rot.to(device), data_rot_label.to(device)

            feature_extractor_output     = feature_extractor(data)
            feature_extractor_output_rot = feature_extractor(data_rot)
            output_rot_output_cat        = torch.cat((feature_extractor_output, feature_extractor_output_rot), dim=1)

            # Should this use inferred labels... ?
            rotation_classifiers, pairs  = get_rotation_classifiers(data_label)
            rot_cls_output               = torch.vstack( [ rot_cls[cls_idx](output_rot_output_cat[data_idx]) for (cls_idx, data_idx) in pairs ], dim=1)

            ground_truths    += data_rot_label
            normality_scores += rot_cls_output

    ground_truths =  torch.tensor([i.item() for i in ground_truths], dtype=int)
    softmax = torch.nn.Softmax(dim=1)
    reshape = lambda x: x.reshape(1, i.size(0))
    normality_scores = torch.vstack( [ softmax(reshape(x)) for i in normality_scores] )
    
    auroc = roc_auc_score(ground_truths.cpu(), normality_scores.cpu(), multi_class='ovr') # 'ovr' or 'ovo' ???
    print('AUROC %.4f' % auroc)

    # create new txt files
    rand = random.randint(0, 100000)
    normality_scores, _ = torch.max(normality_scores, 1)

    if not os.path.isdir('new_txt_list'):
        os.mkdir('new_txt_list')

    # This txt files will have the names of the source images and the names of the target images selected as unknw
    target_unknw = open(f'new_txt_list/{args.source}_known_{str(rand)}.txt', 'w')

    # This txt files will have the names of the target images selected as known
    target_known = open(f'new_txt_list/{args.target}_known_{str(rand)}.txt', 'w')

    known = normality_scores >  args.threshold
    unknw = normality_scores <= args.threshold

    number_of_known_samples = known.sum()
    number_of_unkwn_samples = unknw.sum()

    pairs = zip(target_loader_eval.dataset.names, target_loader_eval.dataset.labels)
    for it, (name, label) in enumerate(files):
        if known[it] > 0:
            target_known.write(f"{name} {str(label)}\n")
        else:
            target_unknw.write(f"{name} {str(label)}\n")
    
    target_known.close()
    target_unknw.close()

    ### Debug Statements
    print(f'# Known: {number_of_known_samples.item()}')
    print(f'# Unknw: {number_of_unkwn_samples.item()}')
    print(f'# Total: {len(labels)}')
    ### Debug Statements

    return rand






