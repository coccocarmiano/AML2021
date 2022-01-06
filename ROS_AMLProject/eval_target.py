
import torch
import numpy as np
import os
from sklearn.metrics import roc_auc_score
import random

#### Implement the evaluation on the target for the known/unknw separation

def evaluation(args,feature_extractor,rot_cls,target_loader_eval,device):

    feature_extractor.eval()
    rot_cls.eval()

    gts, preds, normality_scores = [], [], []

    with torch.no_grad():
        for it, (data, class_l, data_rot, rot_l) in enumerate(target_loader_eval):
            data, class_l, data_rot, rot_l = data.to(device), class_l.to(device), data_rot.to(device), rot_l.to(device)

            feature_extractor_output     = feature_extractor(data)
            feature_extractor_output_rot = feature_extractor(data_rot)
            u = torch.cat((feature_extractor_output, feature_extractor_output_rot), dim=1)
            rot_cls_output = rot_cls(u)

            gts += rot_l
            normality_scores += rot_cls_output

    ground_truths =  torch.tensor([i.item() for i in gts], dtype=int)
    softmax = torch.nn.Softmax(dim=1)
    normality_scores = torch.vstack([softmax(i.reshape(1, i.size(0))) for i in normality_scores])
    
    auroc = roc_auc_score(ground_truths.cpu(), normality_scores.cpu(), multi_class='ovr') # 'ovr' or 'ovo' ???
    print('AUROC %.4f' % auroc)

    # create new txt files
    rand = random.randint(0,100000)
    print('Generated random number is :', rand)

    normality_scores, _ = torch.max(normality_scores, 1)

    if not os.path.isdir('new_txt_list'):
        os.mkdir('new_txt_list')

    # This txt files will have the names of the source images and the names of the target images selected as unknw
    target_unknw = open('new_txt_list/' + args.source + '_known_' + str(rand) + '.txt','w')

    # This txt files will have the names of the target images selected as known
    target_known = open('new_txt_list/' + args.target + '_known_' + str(rand) + '.txt','w')

    known = normality_scores >  args.threshold
    unknw = normality_scores <= args.threshold

    number_of_known_samples = known.sum()
    number_of_unkwn_samples = unknw.sum()

    files  = target_loader_eval.dataset.names
    labels = target_loader_eval.dataset.labels

    for it, name in enumerate(files):
        if known[it] > 0:
            target_known.write(f"{name} {str(labels[it])}\n")
        else:
            target_unknw.write(f"{name} {str(labels[it])}\n")

    files = self.source_loader.dataset.names
    labels = self.source_loader.dataset.labels

    for it, name in enumerate(files):
        target_known.wrtie(f"{name} {str(labels[it])}\n")
    
    target_known.close()
    target_unknw.close()

    print('The number of target samples selected as known is: ', number_of_known_samples.item())
    print('The number of target samples selected as unknw is: ', number_of_unkwn_samples.item())

    return rand






