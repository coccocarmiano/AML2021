import torch
import os
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np


#### Implement the evaluation on the target for the known/unknw separation

def evaluation(args, feature_extractor, rot_cls, obj_cls, get_rotation_classifiers, target_loader_eval, device, rand):
    softmax = torch.nn.Softmax(dim=1)
    feature_extractor.eval()
    obj_cls.eval()

    if args.multihead:
        for head in rot_cls:
            head.eval()
    else:
        rot_cls.eval()

    ground_truths = []
    normality_scores = []

    with torch.no_grad():
        for data, data_label, data_rot, data_rot_label in tqdm(target_loader_eval):
            data, data_label = data.to(device), data_label.to(device)
            data_rot = data_rot.to(device)

            # 1. Extract features from E
            # Extracting original image features from E
            feature_extractor_output = feature_extractor(data)
            # Extracting rotated image features from E
            feature_extractor_output_rot = feature_extractor(data_rot)
            # Concatenate original+rotate features
            feature_extractor_output_conc = torch.cat((feature_extractor_output, feature_extractor_output_rot), dim=1)

            # 2. Get the scores
            # Get the object classifier predictions
            obj_cls_scores = obj_cls(feature_extractor_output)
            predicted_labels = torch.argmax(obj_cls_scores, dim=1)

            # Build a list of rotation classifiers, each belonging to a different sample in the batch
            # If multihead:     R1 will be the head corresponding to the inferred label
            # If not multihead: R1 will be the only existing head
            classifiers = get_rotation_classifiers(predicted_labels)

            # For the center loss version, we need to have both the features coming from the first layer of the discriminator
            # and the output scores coming out from the discriminator
            if args.center_loss:
                # 1. Using the #batch_size discriminators (one for each sample), get the internal features and the output scores
                discriminator_scores = []
                for i in range(len(classifiers)):
                    discriminator_sample_scores, discriminator_sample_features = classifiers[i](feature_extractor_output_conc[i])
                    discriminator_scores.append(discriminator_sample_scores)

                # 2. Transform them back again into tensors (#batch_size, #features_dim)
                discriminator_scores = torch.vstack(discriminator_scores)
            else:
                discriminator_scores = torch.vstack([classifiers[i](feature_extractor_output_conc[i])[0] for i in range(len(classifiers))])

            # Compute softmax and get the maximum probability as the normality score
            r_score = discriminator_scores
            n_scores = softmax(r_score)
            n_score, _ = torch.max(n_scores, dim=1)

            ground_truths.append(data_label.item())
            normality_scores.append(n_score.item())

    ground_truths = np.array(ground_truths, dtype=np.int)
    normality_scores = np.array(normality_scores)

    # Convert to Binary Task : 1 is known, 0 in unknown
    mask_known = ground_truths < 45
    mask_unknw = ground_truths > 44
    ground_truths[mask_known] = 1
    ground_truths[mask_unknw] = 0

    ## Display ROC AUC Value
    auc = roc_auc_score(ground_truths, normality_scores)
    print(f"Computed ROC AUC: {auc:.4f}")

    # Perform the separation using the given threshold
    mask_sep_known = normality_scores >= args.threshold
    mask_sep_unknw = normality_scores < args.threshold

    print(f"Separation performed")
    print(f"Target samples identified as known: {mask_sep_known.sum()} - Actual known samples: {mask_known.sum()}")
    print(f"Target samples identified as known: {mask_sep_unknw.sum()} - Actual unknown samples: {mask_unknw.sum()}")

    ## We now must build and save two datasets
    ## New Source Dataset, with Source + Target Unknown Samples
    ## New Target Dataset, with only Target Known Samples

    if not os.path.isdir('new_txt_list'):
        os.mkdir('new_txt_list')

    # Build new files
    # The new source will contain the original known source plus the unknown target after our separation
    source_and_target_unknown_file = open(f'new_txt_list/{args.source}_known_{str(rand)}.txt', 'w')
    # The new target will contain the known target after our separation
    target_known_file = open(f'new_txt_list/{args.target}_known_{str(rand)}.txt', 'w')

    pairs = zip(target_loader_eval.dataset.names, target_loader_eval.dataset.labels)

    for it, (name, label) in enumerate(pairs):
        if mask_sep_known[it]:
            # Known
            target_known_file.write(f"{name} {str(label)}\n")
        else:
            # Unknown
            source_and_target_unknown_file.write(f"{name} {45}\n")

    source_and_target_unknown_file.close()
    target_known_file.close()
