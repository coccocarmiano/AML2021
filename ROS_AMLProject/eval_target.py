
from tqdm import tqdm
import torch
import numpy as np
import os
from sklearn.metrics import roc_auc_score
import random
from tqdm import tqdm

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
    
    ground_truths, normality_scores = [], []

    with torch.no_grad():
        for data, data_label, data_rot, data_rot_label in tqdm(target_loader_eval):
            data,     data_label     =  data.to(device),     data_label.to(device)
            data_rot, data_rot_label =  data_rot.to(device), data_rot_label.to(device)

            data_label[data_label > 44] = 45

            feature_extractor_output     = feature_extractor(data)
            feature_extractor_output_rot = feature_extractor(data_rot)
            output_rot_output_cat        = torch.cat((feature_extractor_output, feature_extractor_output_rot), dim=1)

            # TODO: Should this use inferred labels... (silvia said so) ?
            rotation_classifiers         = get_rotation_classifiers(data_label)
            it = range(len(rotation_classifiers))
            rot_cls_output               = torch.vstack( [ rotation_classifiers[idx](output_rot_output_cat[idx]) for idx in it ])
            rot_cls_output_softmax = softmax(rot_cls_output)
            r_preds, _ = torch.max(rot_cls_output_softmax, dim=1)
            normality_scores.append(r_preds.item())
            #ground_truths    += data_rot_label
            #normality_scores += rot_cls_output

    #ground_truths =  torch.tensor([i.item() for i in ground_truths], dtype=int)
    #softmax = torch.nn.Softmax(dim=1)
    #reshape = lambda x: x.reshape(1, x.size(0))
    #normality_scores = torch.vstack( [ softmax(reshape(i)) for i in normality_scores] )

    # Build ground truths
    target_known_f = open('txt_list/' + args.target + '_known.txt', 'r')
    known_file_names = target_known_f.readlines()
    known_file_names = [l.strip() for l in known_file_names]

    gts = []
    file_names = target_loader_eval.dataset.names
    labels = target_loader_eval.dataset.labels
    for n, l in zip(file_names, labels):
        if f"{n} {l}" in known_file_names:
            gts.append(1)
        else:
            gts.append(0)

    target_known_f.close()

    # ground_truths =  torch.tensor([i.item() for i in gts], dtype=int)
    # softmax = torch.nn.Softmax(dim=1)
    # normality_scores = torch.vstack([softmax(i.reshape(1, i.size(0))) for i in normality_scores])

    # auroc = roc_auc_score(ground_truths.cpu(), normality_scores.cpu(), multi_class='ovr') # 'ovr' or 'ovo' ???
    print(f"gts len: {len(gts)}")
    print(f"gts: {gts}")
    print(f"normality scores len: {len(normality_scores)}")
    print(f"normality scores: {normality_scores}")
    normality_scores = np.array(normality_scores)
    auroc = roc_auc_score(gts, normality_scores)
    print('AUROC %.4f' % auroc)

    # TODO: Should it test the goodness of R1 in predicting rotations or in separating the samples in known and unknown?
    #auroc = roc_auc_score(ground_truths.cpu(), normality_scores.cpu(), multi_class='ovr') # 'ovr' or 'ovo' ???
    #print('AUROC %.4f' % auroc)

    # create new txt files
    rand = random.randint(0, 100000)
    #normality_scores, _ = torch.max(normality_scores, 1)

    #normality_scores, _ = torch.max(normality_scores, 1)

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
    for it, (name, label) in enumerate(pairs):
        if known[it] > 0:
            target_known.write(f"{name} {str(label)}\n")
        else:
            target_unknw.write(f"{name} {str(label)}\n")
    
    target_known.close()
    target_unknw.close()

    return rand






