import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler
from center_loss import CenterLoss
import matplotlib.pyplot as plt

from tqdm import tqdm

#### Implement Step1

def _do_epoch(args, E, C, R, source_loader, device, optimizer, optimizer_CL=None, criterion_CL=None):

    C_criterion = nn.CrossEntropyLoss()
    R_criterion = nn.CrossEntropyLoss()

    C_correct_preds, R_correct_preds, C_tot_preds, R_tot_preds = 0, 0, 0, 0
    tot_avg_loss, C_avg_loss, R_avg_loss, CL_avg_loss = 0.0, 0.0, 0.0, 0.0
    tot_batches = 0
    for batch_samples, batch_labels, batch_samples_rot, batch_labels_rot in tqdm(source_loader):
        tot_batches += 1
        # Zero-out gradients
        # Move data to GPU
        batch_samples    , batch_labels     = batch_samples.to(device)    , batch_labels.to(device),
        batch_samples_rot, batch_labels_rot = batch_samples_rot.to(device), batch_labels_rot.to(device)

        optimizer.zero_grad()
        if args.center_loss:
            optimizer_CL.zero_grad()


        # 1. Extract features from E
        # Extracting original image features from E
        E_output     = E(batch_samples)
        # Extracting rotated image features from E
        E_output_rot = E(batch_samples_rot)
        # Concatenate original+rotate features
        E_output_conc = torch.cat((E_output, E_output_rot), dim=1)

        # 2. Get the scores
        # Use C1 to get the scores (without softmax, the cross-entropy will do it)
        C_scores = C(E_output)

        # Use R1 to get the scores
        # For the center loss version, we need to have both the features coming from the first layer of the discriminator
        # and the output scores coming out from the discriminator
        if args.center_loss:
            R_features, R_scores = R.forward_extended(E_output_conc)
        else:
            R_scores = R(E_output_conc)

        C_loss  = C_criterion(C_scores, batch_labels)
        R_loss    = R_criterion(R_scores, batch_labels_rot) * args.weight_RotTask_step1
        CL_loss   = .0

        if args.center_loss:
            CL_loss = criterion_CL(R_features, batch_labels_rot) * args.weight_CL

        loss = C_loss + R_loss + CL_loss

        tot_avg_loss += loss.data.item()
        C_avg_loss += C_loss.data.item()
        R_avg_loss += R_loss.data.item()
        CL_avg_loss += CL_loss.data.item()

        # 3. Compute total loss, backward, step, etc
        loss.backward()
        optimizer.step()

        # by doing so, weight_CL would not impact on the learning of centers
        if args.center_loss:
            for param in criterion_CL.parameters():
                param.grad.data *= (1. / args.weight_CL)
            optimizer_CL.step()

        # Extract class predictions
        preds        = torch.argmax(C_scores, dim=1)
        C_correct_preds += (preds == batch_labels).sum().item()

        # Extract rot predictions
        preds        = torch.argmax(R_scores, dim=1)
        R_correct_preds += (preds == batch_labels_rot).sum().item()

        C_tot_preds     += batch_labels.size(0)
        R_tot_preds     += batch_labels_rot.size(0)

    tot_avg_loss /= tot_batches
    C_avg_loss /= tot_batches
    R_avg_loss /= tot_batches
    CL_avg_loss /= tot_batches

    C_accuracy = C_correct_preds / C_tot_preds
    R_accuracy = R_correct_preds / R_tot_preds

    return tot_avg_loss, C_avg_loss, R_avg_loss, CL_avg_loss, C_accuracy, R_accuracy


def step1(args, E, C, R, source_loader, device, optimizer, scheduler, optimizer_CL=None, scheduler_CL=None, criterion_CL=None):
    # Set the training mode
    E.train()
    C.train()
    R.train()

    E.to(device)
    C.to(device)
    R.to(device)

    history = {}
    history['tot_loss'] = []
    history['C_loss'] = []
    history['R_loss'] = []
    history['CL_loss'] = []
    history['C_acc'] = []
    history['R_acc'] = []

    for epoch in range(args.epochs_step1):
        print(f'Epoch {epoch+1}/{args.epochs_step1}')
        tot_loss, C_loss, R_loss, CL_loss, C_accuracy, R_accuracy = _do_epoch(args, E, C, R, source_loader, device, optimizer, optimizer_CL=optimizer_CL, criterion_CL=criterion_CL)
        print(f"\tTotal Loss: {tot_loss:.4f}")
        print(f"\tClass Loss: {C_loss:.4f}")
        print(f"\tRot Loss: {R_loss:.4f}")
        if args.center_loss:
            print(f"\tCenterLoss: {CL_loss:.4f}")
        print(f"\tClass Accuracy: {C_accuracy*100:.2f} %")
        print(f"\tRot Accuracy: {R_accuracy*100:.2f} %")

        history['tot_loss'].append(tot_loss)
        history['C_loss'].append(C_loss)
        history['R_loss'].append(R_loss)
        history['CL_loss'].append(CL_loss)
        history['C_acc'].append(C_accuracy)
        history['R_acc'].append(R_accuracy)

        scheduler.step()
        if args.center_loss:
            scheduler_CL.step()

    return history

def plot_step1_accuracy_loss(args, history):
    tot_loss = history['tot_loss']
    C_loss = history['C_loss']
    R_loss = history['R_loss']
    CL_loss = history['CL_loss']
    C_accuracy = history['C_acc']
    R_accuracy = history['R_acc']

    epochs = range(1, len(tot_loss) + 1)

    # Accuracy plot
    plt.figure()
    plt.title('Object classifier and rotation classifier accuracy over step1 epochs')
    plt.plot(epochs, C_accuracy, 'b', label='Object classifier accuracy')
    plt.plot(epochs, R_accuracy, 'r', label='Rotation classifier accuracy')
    plt.legend()

    # Loss plot
    plt.figure()
    plt.title('Object classifier and rotation classifier losses over step1 epochs')
    plt.plot(epochs, tot_loss, 'm', label='Total classifier loss')
    plt.plot(epochs, C_loss, 'b', label='Object classifier loss')
    plt.plot(epochs, R_loss, 'r', label='Rotation classifier loss')
    if args.center_loss:
        plt.plot(epochs, CL_loss, 'g', label='Center Loss')

    plt.legend()

    plt.show()

