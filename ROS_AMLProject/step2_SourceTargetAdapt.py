import torch
from torch import nn
from eval_target import target_evaluation
from itertools import cycle
from tqdm import tqdm
import matplotlib.pyplot as plt

def _do_epoch(args, E, C, R, source_loader, target_loader_train, optimizer, device):

    # Double the learning rate for the unknown class
    # w = [ 1.0 for _ in range(args.n_classes_known + 1) ]
    # w[-1] = 2.0
    # w = torch.tensor(w).to(device)

    # C_criterion = nn.CrossEntropyLoss(weight=w)
    C_criterion = nn.CrossEntropyLoss()
    R_criterion = nn.CrossEntropyLoss()

    C_correct_preds, R_correct_preds, C_tot_preds, R_tot_preds = 0, 0, 0, 0
    tot_avg_loss, C_avg_loss, R_avg_loss = 0.0, 0.0, 0.0

    # We iterate over two datasets at once
    target_loader_train = cycle(target_loader_train)

    tot_known, tot_unknown = 0, 0
    tot_predicted_known, tot_predicted_unknown = 0, 0

    # tot_rot = [0, 0, 0, 0]
    # tot_pred_rot = [0, 0, 0, 0]

    tot_batches = 0
    for source_batch_samples, source_batch_labels, _, _ in tqdm(source_loader):
        (target_batch_samples, _, target_batch_samples_rot, target_batch_labels_rot) = next(target_loader_train)
        tot_batches += 1

        # Zero-out the gradients
        optimizer.zero_grad()

        # Move everything to the used device ( CPU/GPU )
        source_batch_samples = source_batch_samples.to(device)
        source_batch_labels = source_batch_labels.to(device)
        target_batch_samples = target_batch_samples.to(device)
        target_batch_samples_rot = target_batch_samples_rot.to(device)
        target_batch_labels_rot = target_batch_labels_rot.to(device)

        # 1. Extract features from E
        # Extracting source image features from E
        E_source_output     = E(source_batch_samples)
        # Extracting original target image features from E
        E_target_output     = E(target_batch_samples)
        # Extracting rotated target image features from E
        E_target_output_rot = E(target_batch_samples_rot)
        # Concatenate original+rotate target features
        E_target_output_conc = torch.cat((E_target_output, E_target_output_rot), dim=1)

        # 2. Get the scores
        # For the source images, we get the output scores from C2
        C_source_scores = C(E_source_output)

        # For the target image, we want the scores from R2
        R_scores = R(E_target_output_conc)

        # Evaluate losses
        C_loss  = C_criterion(C_source_scores, source_batch_labels)
        R_loss    = R_criterion(R_scores, target_batch_labels_rot) * args.weight_RotTask_step2
        loss = C_loss + R_loss

        tot_avg_loss += loss.data.item()
        C_avg_loss += C_loss.data.item()
        R_avg_loss += R_loss.data.item()

        cls_pred     = torch.argmax(C_source_scores, dim=1)
        C_correct_preds += (cls_pred == source_batch_labels).sum().item()

        rot_pred     = torch.argmax(R_scores, dim=1)
        R_correct_preds += (rot_pred == target_batch_labels_rot).sum().item()

        loss.backward()
        optimizer.step()

        C_tot_preds += source_batch_labels.size(0)
        R_tot_preds += target_batch_labels_rot.size(0)

        # debug
        tot_known += (source_batch_labels < 45).sum().item()
        tot_unknown += (source_batch_labels > 44).sum().item()
        tot_predicted_known += (cls_pred < 45).sum().item()
        tot_predicted_unknown += (cls_pred > 44).sum().item()
        # for blr in target_batch_labels_rot:
        #     tot_rot[blr] += 1
        # for rp in rot_pred:
        #     tot_pred_rot[rp] += 1


    print(f"Training - tot known: {tot_known}")
    print(f"Training - tot unknown: {tot_unknown}")
    print(f"Training - tot predicted known: {tot_predicted_known}")
    print(f"Training - tot predicted unknown: {tot_predicted_unknown}")
    # for i in range(4):
    #     print(f"Training - tot rot with label {i}: {tot_rot[i]}")
    # for i in range(4):
    #     print(f"Training - tot predicted rot with label {i}: {tot_pred_rot[i]}")

    tot_avg_loss /= tot_batches
    C_avg_loss /= tot_batches
    R_avg_loss /= tot_batches
    C_accuracy = C_correct_preds / C_tot_preds
    R_accuracy = R_correct_preds / R_tot_preds

    return tot_avg_loss, C_avg_loss, R_avg_loss, C_accuracy, R_accuracy


def step2(args, E, C, R, source_loader, target_loader_train, target_loader_eval, device, optimizer, scheduler):
    """
    Performs the domain alignment through R2 while learning the unknown class using C2.
    After each epoch, we evaluate the performance of C2 on the evaluation target dataset.
    """

    E = E.to(device)
    C = C.to(device)
    R = R.to(device)

    history = {}
    history['tot_loss'] = []
    history['C_loss'] = []
    history['R_loss'] = []
    history['C_acc'] = []
    history['R_acc'] = []
    history['eval_HOS'] = []
    history['eval_OS'] = []
    history['eval_UNK'] = []
    history['eval_C_loss'] = []

    for epoch in range(args.epochs_step2):
        print(f"Epoch {epoch+1}/{args.epochs_step2}")

        # TRAINING EPOCH
        # Set the training mode
        E.train()
        C.train()
        R.train()

        tot_loss, C_loss, R_loss, C_accuracy, R_accuracy = _do_epoch(args, E, C, R, source_loader, target_loader_train, optimizer, device)
        history['tot_loss'].append(tot_loss)
        history['C_loss'].append(C_loss)
        history['R_loss'].append(R_loss)
        history['C_acc'].append(C_accuracy)
        history['R_acc'].append(R_accuracy)

        scheduler.step()

        print("Train Stats:")
        print(f"\tTotal Loss: {tot_loss:.4f}")
        print(f"\tClass Loss: {C_loss:.4f}")
        print(f"\tRot Loss: {R_loss:.4f}")
        print(f"\tClass Accuracy: {C_accuracy * 100:.2f} %")
        print(f"\tRot Accuracy: {R_accuracy * 100:.2f} %")
        print()

        # EVAL EPOCH
        HOS, OS, UNK, C_loss = target_evaluation(args, E, C, target_loader_eval, device)
        history['eval_HOS'].append(HOS)
        history['eval_OS'].append(OS)
        history['eval_UNK'].append(UNK)
        history['eval_C_loss'].append(C_loss)

        print()
        print("Target Evaluation Stats")
        print(f"\tClass Loss: {C_loss:.2f}")
        print(f"\tOS : {OS * 100:.2f} %")
        print(f"\tUNK: {UNK * 100:.2f} %")
        print(f"\tHOS: {HOS * 100:.2f} %")

    return history

def plot_step2_accuracy_loss(args, history):
    tot_loss = history['tot_loss']
    C_loss = history['C_loss']
    R_loss = history['R_loss']
    C_accuracy = history['C_acc']
    R_accuracy = history['R_acc']

    epochs = range(1, len(tot_loss) + 1)

    # Accuracy plot
    plt.figure()
    plt.title('(Training) Object classifier and rotation classifier accuracy over step2 epochs')
    plt.plot(epochs, C_accuracy, 'b', label='Object classifier accuracy')
    plt.plot(epochs, R_accuracy, 'r', label='Rotation classifier accuracy')
    plt.legend()

    # Loss plot
    plt.figure()
    plt.title('(Training) Object classifier and rotation classifier losses over step2 epochs')
    plt.plot(epochs, tot_loss, 'm', label='Total loss')
    plt.plot(epochs, C_loss, 'b', label='Object classifier loss')
    plt.plot(epochs, R_loss, 'r', label='Rotation classifier loss')
    plt.legend()

    plt.show()

def plot_eval_performance(args, history):
    HOS = history['eval_HOS']
    OS = history['eval_OS']
    UNK = history['eval_UNK']
    C_loss = history['eval_C_loss']

    epochs = range(1, len(HOS) + 1)

    # Accuracy plot
    plt.figure()
    plt.title(f'Evaluation - HOS, OS, UNK over the target dataset (MH: {args.multihead} - CL: {args.center_loss})')
    plt.plot(epochs, HOS, 'b', label='HOS')
    plt.plot(epochs, OS, 'r', label='OS')
    plt.plot(epochs, UNK, 'm', label='UNK')
    plt.legend()

    # Loss plot
    plt.figure()
    plt.title(f'Evaluation - Object classifier loss (MH: {args.multihead} - CL: {args.center_loss})')
    plt.plot(epochs, C_loss, 'b', label='Object classifier loss')
    plt.legend()

    plt.show()