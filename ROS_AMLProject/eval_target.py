
import torch
import numpy as np
import os
from sklearn.metrics import roc_auc_score
import random
import pickle


#### Implement the evaluation on the target for the known/unknown separation

def evaluation(args,feature_extractor,rot_cls,target_loader_eval,device):
    softmax = torch.nn.Softmax(dim=1)
    feature_extractor.eval()
    rot_cls.eval()

    #gts, preds, normality_scores = [], [], []
    normality_scores = []

    with torch.no_grad():
        for it, (data,class_l,data_rot,rot_l) in enumerate(target_loader_eval):
            data, class_l, data_rot, rot_l = data.to(device), class_l.to(device), data_rot.to(device), rot_l.to(device)

            feature_extractor_output     = feature_extractor(data)
            feature_extractor_output_rot = feature_extractor(data_rot)
            u = torch.cat((feature_extractor_output, feature_extractor_output_rot), dim=1)
            rot_cls_output = rot_cls(u)
            rot_cls_output_softmax = softmax(rot_cls_output)
            # r_preds = torch.argmax(rot_cls_output, dim=1)
            # r_preds = rot_cls_output
            r_preds, _ = torch.max(rot_cls_output_softmax, dim=1)

            #gts += rot_l
            normality_scores += r_preds.item()
    
#     with open("gts", "wb") as gtsf:
#         pickle.dump(gts, gtsf)
#     
#     with open("normality_scores", "wb") as normality_scoresf:
#         pickle.dump(normality_scores, normality_scoresf)
#         
# def evaluation2(args,feature_extractor,rot_cls,target_loader_eval,device):
# 
#     with open("gts", "rb") as gtsf:
#         gts = pickle.load(gtsf)
# 
#     with open("normality_scores", "rb") as normality_scoresf:
#         normality_scores = pickle.load(normality_scoresf)

    # Build ground truths
    target_known_f = open('txt_list/' + args.target + '_known.txt','r')
    known_file_names = target_known_f.readlines()

    gts = []
    file_names = target_loader_eval.dataset.names
    labels = target_loader_eval.dataset.labels
    for n, l in zip(file_names, labels):
        if f"{n} {l}" in known_file_names:
            gts.append(1)
        else:
            gts.append(0)

    target_known_f.close()

    #ground_truths =  torch.tensor([i.item() for i in gts], dtype=int)
    #softmax = torch.nn.Softmax(dim=1)
    #normality_scores = torch.vstack([softmax(i.reshape(1, i.size(0))) for i in normality_scores])
    
    #auroc = roc_auc_score(ground_truths.cpu(), normality_scores.cpu(), multi_class='ovr') # 'ovr' or 'ovo' ???
    print(f"gts len: {len(gts)}")
    print(f"gts: {gts}")
    print(f"normality scores len: {len(normality_scores)}")
    print(f"normality scores: {normality_scores}")
    auroc = roc_auc_score(gts, normality_scores)
    print('AUROC %.4f' % auroc)

    # create new txt files
    rand = random.randint(0,100000)
    print('Generated random number is :', rand)

    #normality_scores, _ = torch.max(normality_scores, 1)

    if not os.path.isdir('new_txt_list'):
        os.mkdir('new_txt_list')

    # This txt files will have the names of the source images and the names of the target images selected as unknown
    #target_unknown = open('new_txt_list/' + args.source + '_known_' + str(rand) + '.txt','w')

    # This txt files will have the names of the target images selected as known
    target_known = open('new_txt_list/' + args.target + '_known_' + str(rand) + '.txt','w')

    normality_scores = np.array(normality_scores)
    known = normality_scores > args.threshold
    unknown = normality_scores <= args.threshold

    number_of_known_samples = known.sum()
    number_of_unknown_samples = unknown.sum()
    #files = target_loader_eval.dataset.names
    #labels = target_loader_eval.dataset.labels

    for it, name in enumerate(file_names):
        if known[it] > 0:
            target_known.write(name + ' ' + str(labels[it]) + '\n')
        #else:
        #    target_unknown.write(name + ' ' + str(labels[it]) + '\n')
        
    target_known.close()
    #target_unknown.close()

    print('The number of target samples selected as known is: ',number_of_known_samples)
    print('The number of target samples selected as unknown is: ', number_of_unknown_samples)

    return rand






