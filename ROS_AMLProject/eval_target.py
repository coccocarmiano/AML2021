import torch
import os
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
from torch import nn

softmax = torch.nn.Softmax(dim=1)

def target_separation(args, E, C, R, target_loader_eval, device, rand):
    """
    Separate the target into known and unknown and generate the new source and target files.
    New source file: Original known source + Unknown target according to the performed separation
    New target file: Unknown target according to the performed separation
    """

    E.eval()
    C.eval()
    R.custom_eval()

    E.to(device)
    C.to(device)
    R.custom_to(device)

    ground_truths = []
    normality_scores = []

    with torch.no_grad():
        for batch_samples, batch_labels, batch_samples_rot, batch_labels_rot in tqdm(target_loader_eval):
            batch_samples, batch_labels = batch_samples.to(device), batch_labels.to(device)
            batch_samples_rot = batch_samples_rot.to(device)

            # 1. Extract features from E
            # Extracting original image features from E
            E_output = E(batch_samples)
            # Extracting rotated image features from E
            E_output_rot = E(batch_samples_rot)
            # Concatenate original+rotate features
            E_output_conc = torch.cat((E_output, E_output_rot), dim=1)

            # 2. Get the scores
            obj_cls_scores = C(E_output)
            predicted_labels = torch.argmax(obj_cls_scores, dim=1)

            # Use R1 to get the scores
            R_scores = R(E_output_conc)

            # TODO: select only one head or apply the softmax to the whole set of output scores from all the heads?
            # If R1 is multihead, we pick the head corresponding to the inferred label
            if args.multihead:
                mask = []
                for sample_label in predicted_labels:
                    sample_mask = list(range(sample_label * 4, sample_label * 4 + 4))
                    mask.append(sample_mask)
                mask = torch.tensor(mask).to(device)
                R_scores = R_scores[mask]
                print(f"Debug - R_scores with multihead: {R_scores.size()}")

            # Compute softmax and get the maximum probability as the normality score
            R_probabilities = softmax(R_scores)
            n_score, _ = torch.max(R_probabilities, dim=1)

            ground_truths.append(batch_labels.item())
            normality_scores.append(n_score.item())

    ground_truths = np.array(ground_truths, dtype=np.int)
    normality_scores = np.array(normality_scores)

    # Convert to Binary Task : 1 is known, 0 in unknown
    mask_known = ground_truths < 45
    mask_unknw = ground_truths > 44
    ground_truths[mask_known] = 1
    ground_truths[mask_unknw] = 0

    # Compute AUC-ROC value
    auc = roc_auc_score(ground_truths, normality_scores)
    print("\n")
    print(f"Computed ROC AUC: {auc:.4f}")
    print()

    # Perform the separation using the given threshold
    mask_sep_known = normality_scores >= args.threshold
    mask_sep_unknw = normality_scores < args.threshold

    print(f"Separation performed using threshold: {args.threshold:.3f}")
    print(f"Target samples identified as known: {mask_sep_known.sum()} - Actual known samples: {mask_known.sum()}")
    print(f"Target samples identified as unknown: {mask_sep_unknw.sum()} - Actual unknown samples: {mask_unknw.sum()}")

    known_accuracy = (mask_sep_known == mask_known).sum() / mask_sep_known.shape[0]
    # unk_accuracy = (mask_sep_unknw == mask_unknw).sum() / mask_sep_unknw.shape[0]

    print(f"Separation accuracy: {known_accuracy*100:.2f} %")
    #print(f"Unknown separation accuracy: {unk_accuracy*100:.2f} %")
    print()

    # We now must build and save two datasets
    # New Source Dataset, with Source + Target Unknown Samples
    # New Target Dataset, with only Target Known Samples

    if not os.path.isdir('new_txt_list'):
        os.mkdir('new_txt_list')

    # Build new files
    # The new source will contain the original known source plus the unknown target after our separation
    stu_fname = f'new_txt_list/{args.source}_known_{str(rand)}.txt'
    source_and_target_unknown_file = open(stu_fname, 'w')
    # The new target will contain the known target after our separation
    tk_fname = f'new_txt_list/{args.target}_known_{str(rand)}.txt'
    target_known_file = open(tk_fname, 'w')

    pairs = zip(target_loader_eval.dataset.names, target_loader_eval.dataset.labels)

    for it, (name, label) in enumerate(pairs):
        if mask_sep_known[it]:
            # Known
            target_known_file.write(f"{name} {-1}\n")
        else:
            # Unknown
            source_and_target_unknown_file.write(f"{name} {45}\n")

    source_and_target_unknown_file.close()
    target_known_file.close()

    print(f"New source file containing known source and unknown target (according to the separation) written in {stu_fname}")
    print(f"New target file containing known target (according to the separation) written in {tk_fname}")

    return auc, known_accuracy

def target_evaluation(args, E, C, R, target_loader_eval, device):
    # Final evaluation on the target using only C2
    # Disable training
    E.eval()
    C.eval()
    R.custom_eval()

    E.to(device)
    C.to(device)
    R.custom_to(device)


    C_criterion = nn.CrossEntropyLoss()

    tot_known, tot_unkwn, known_correct, unknw_correct = 0, 0, 0, 0
    C_avg_loss = 0.0
    tot_batches = 0
    with torch.no_grad():
        for batch_samples, batch_labels, _, _ in tqdm(target_loader_eval):
            tot_batches += 1
            batch_samples, batch_labels = batch_samples.to(device), batch_labels.to(device)

            E_output = E(batch_samples)
            C_scores = C(E_output)
            C_preds = torch.argmax(C_scores, dim=1)

            C_loss = C_criterion(C_scores, batch_labels)
            C_avg_loss += C_loss.data.item()

            known_mask = batch_labels < 45
            unknw_mask = batch_labels > 44

            batch_labels[unknw_mask] = 45

            tot_known += known_mask.sum().item()
            tot_unkwn += unknw_mask.sum().item()

            known_correct += (C_preds[known_mask] == batch_labels[known_mask]).sum().item()
            unknw_correct += (C_preds[unknw_mask] == batch_labels[unknw_mask]).sum().item()

        C_avg_loss /= tot_batches
        OS = known_correct / tot_known
        UNK = unknw_correct / tot_unkwn
        HOS = 2 / (1.0 / float(OS) + 1.0 / float(UNK))

        return HOS, OS, UNK, C_avg_loss