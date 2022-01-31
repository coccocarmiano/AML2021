import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler
from center_loss import CenterLoss

from tqdm import tqdm

#### Implement Step1

def _do_epoch(args, feature_extractor, rot_cls, obj_cls, get_rotation_classifiers, source_loader, optimizer, device):

    cls_criterion = nn.CrossEntropyLoss()
    rot_criterion_ce = nn.CrossEntropyLoss()

    if args.center_loss:
        criterion_center = CenterLoss(num_classes=4, feat_dim=256, use_gpu=True, device=device) #version 2: features from first layer of R1
        optimizer_center = torch.optim.SGD(criterion_center.parameters(), lr=args.learning_rate_center) #version a: used a specified LR for center loss

    feature_extractor.train()
    obj_cls.train()

    if args.multihead:
        for head in rot_cls:
            head.train()
    else:
        rot_cls.train()

    cls_correct, rot_correct, cls_tot, rot_tot = 0, 0, 0, 0

    for data, data_label, data_rot, data_rot_label in tqdm(source_loader):
        
        # Zero-out gradients
        optimizer.zero_grad()
        if args.center_loss:
            optimizer_center.zero_grad()

        # Move data to GPU
        data    , data_label     = data.to(device)    , data_label.to(device),
        data_rot, data_rot_label = data_rot.to(device), data_rot_label.to(device)

        # 1. Extract features from E
        # Extracting original image features from E
        feature_extractor_output     = feature_extractor(data)
        # Extracting rotated image features from E
        feature_extractor_output_rot = feature_extractor(data_rot)
        # Concatenate original+rotate features
        feature_extractor_output_conc = torch.cat((feature_extractor_output, feature_extractor_output_rot), dim=1)

        # 2. Get the scores
        # Use C1 to get the scores (without softmax, the cross-entropy will do it)
        obj_cls_output = obj_cls(feature_extractor_output)

        # Build a list of rotation classifiers, each belonging to a different sample in the batch
        # If multihead:     R1 will be the head corresponding to the groundtruth label
        # If not multihead: R1 will be the only existing head
        classifiers    = get_rotation_classifiers(data_label)

        # For the center loss version, we need to have both the features coming from the first layer of the discriminator
        # and the output scores coming out from the discriminator
        if args.center_loss:  
            # 1. Using the #batch_size discriminators (one for each sample), get the internal features and the output scores
            discriminator_features = []
            discriminator_scores = []
            for i in range(len(classifiers)):
                discriminator_sample_scores, discriminator_sample_features = classifiers[i](feature_extractor_output_conc[i])
                discriminator_features.append(discriminator_sample_features)
                discriminator_scores.append(discriminator_sample_scores)

            # 2. Transform them back again into tensors (#batch_size, #features_dim)
            discriminator_features = torch.vstack(discriminator_features)
            discriminator_scores = torch.vstack(discriminator_scores)
        else:
            discriminator_scores = torch.vstack([classifiers[i](feature_extractor_output_conc[i])[0] for i in range(len(classifiers))])

        class_loss  = cls_criterion(obj_cls_output, data_label)
        rot_loss    = rot_criterion_ce(discriminator_scores, data_rot_label) * args.weight_RotTask_step1
        cent_loss   = .0

        if args.center_loss:
            # Version 1: Use `feature_extra` as input for Center Loss
            #cent_loss  = criterion_center(rot_cls_output, data_rot_label) * args.cl_lambda  #version 1: features from feature extractor

            #Version 2: Use `Discriminator` classifier output as input for Center Loss
            cent_loss = criterion_center(discriminator_features, data_rot_label) * args.cl_lambda


        # 3. Compute total loss, backward, step, etc
        loss = class_loss + rot_loss + cent_loss
        loss.backward()
        optimizer.step()
        # by doing so, cl_lambda would not impact on the learning of centers
        if args.center_loss:
            for param in criterion_center.parameters():
                param.grad.data *= (1. / args.cl_lambda)
            optimizer_center.step()

        # Extract class predictions
        preds        = torch.argmax(obj_cls_output, dim=1)
        cls_correct += (preds == data_label).sum().item()

        # Extract rot predictions
        preds        = torch.argmax(discriminator_scores, dim=1)
        rot_correct += (preds == data_rot_label).sum().item()

        cls_tot     += data_label.size(0)
        rot_tot     += data_rot_label.size(0)


    acc_cls = cls_correct / cls_tot
    acc_rot = rot_correct / rot_tot

    return class_loss, acc_cls, rot_loss, cent_loss, acc_rot


def step1(args, feature_extractor, rot_cls, obj_cls, get_rotation_classifiers, source_loader, device):
    optimizer, scheduler = get_optim_and_scheduler(feature_extractor, rot_cls, obj_cls, args.epochs_step1, args.learning_rate, args.train_all, args.multihead)
    feature_extractor.train()
    obj_cls.train()

    if args.multihead:
        for head in rot_cls:
            head.train()
    else:
        rot_cls.train()

    for epoch in range(args.epochs_step1):
        print(f'Epoch {epoch+1}/{args.epochs_step1}')
        class_loss, acc_cls, loss_ce, loss_cl, acc_rot = _do_epoch(args, feature_extractor, rot_cls, obj_cls, get_rotation_classifiers, source_loader, optimizer, device)
        print(f"\tClass Loss    : {class_loss.item():.4f}")
        print(f"\tRot   Loss    : {loss_ce.item():.4f}")
        if args.center_loss:
            print(f"\tCenterLoss    : {loss_cl.item():.4f}")
        print(f"\tClass Accuracy: {acc_cls*100:.2f}%")
        print(f"\tRot   Accuracy: {acc_rot*100:.2f}%")
        scheduler.step()
