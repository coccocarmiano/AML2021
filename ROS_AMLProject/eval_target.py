
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import random
import numpy as np


#### Implement the evaluation on the target for the known/unknown separation

def evaluation(args,feature_extractor,rot_cls,target_loader_eval,device):

    feature_extractor.eval()
    rot_cls.eval()

    gts, preds, normality_scores = [], [], []

    with torch.no_grad():
        for it, (data,class_l,data_rot,rot_l) in enumerate(target_loader_eval):
            data, class_l, data_rot, rot_l = data.to(device), class_l.to(device), data_rot.to(device), rot_l.to(device)

            feature_extractor_output     = feature_extractor(data)
            feature_extractor_output_rot = feature_extractor(data_rot)
            u = torch.cat((feature_extractor_output, feature_extractor_output_rot), dim=1)
            rot_cls_output = rot_cls(u)
            # r_preds = torch.argmax(rot_cls_output, dim=1)
            r_preds = rot_cls_output

            gts += rot_l
            normality_scores += r_preds

    
    ground_truths =  torch.tensor([i.item() for i in gts], dtype=int)
    normality_scores = torch.cat(normality_scores)
    auroc = roc_auc_score(ground_truths, normality_scores, multi_class='ovr') # 'ovr' or 'ovo' ???
    print('AUROC %.4f' % auroc)

    # create new txt files
    rand = random.randint(0,100000)
    print('Generated random number is :', rand)

    # This txt files will have the names of the source images and the names of the target images selected as unknown
    target_unknown = open('new_txt_list/' + args.source + '_known_' + str(rand) + '.txt','w')

    # This txt files will have the names of the target images selected as known
    target_known = open('new_txt_list/' + args.target + '_known_' + str(rand) + '.txt','w')

    known = normality_score > args.threshold
    unknown = normality_score <= args.threshold

    number_of_known_samples = known.sum()
    number_of_unknown_samples = unknown.sum()
    files = target_loader_eval.dataset.names
    labels = target_loader_eval.dataset.labels

    for it, name in enumerate(files):
        if known[it]:
            target_known.write(name + ' ' + str(labels[it]) + '\n')
        else:
            target_unknown.write(name + ' ' + str(labels[it]) + '\n')
        
    target_known.close()
    target_unknown.close()

    print('The number of target samples selected as known is: ',number_of_known_samples)
    print('The number of target samples selected as unknown is: ', number_of_unknown_samples)

    return rand






