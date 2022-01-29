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
        
        # Extract Features
        feature_extractor_output     = feature_extractor(data)
        feature_extractor_output_rot = feature_extractor(data_rot)

        # Acquire scores for C1/R1
        obj_cls_output = obj_cls(feature_extractor_output)
        rot_cls_output = torch.cat((feature_extractor_output, feature_extractor_output_rot), dim=1)

        # Get the classifiers that shall predict the rotation and how many they are
        classifiers    = get_rotation_classifiers(data_label)
        it             = range(len(classifiers))

        if args.center_loss:  

            # First we get a `zip` of both outputs and features
            output_and_feats = [ classifiers[idx](rot_cls_output[idx]) for idx in it ]

            # Then we unzip it into two tuples and covert them into two lists and then into tensors
            rot_cls_output, features = zip(*output_and_feats) 

            rot_cls_output = list(rot_cls_output)
            features       = list(features)

            features = torch.vstack(features)                   
            rot_cls_output = torch.vstack(rot_cls_output)

        else:
            rot_cls_output = torch.vstack([classifiers[idx](rot_cls_output[idx]) for idx in it])   

        class_loss  = cls_criterion(obj_cls_output, data_label)
        rot_loss    = rot_criterion_ce(rot_cls_output, data_rot_label) * args.weight_RotTask_step1
        cent_loss   = .0

        if args.center_loss:
            # Version 1: Use `feature_extra` as input for Center Loss
            #cent_loss  = criterion_center(rot_cls_output, data_rot_label) * args.cl_lambda  #version 1: features from feature extractor

            #Version 2: Use `Discriminator` classifier output as input for Center Loss
            cent_loss = criterion_center(features, data_rot_label) * args.cl_lambda


        # Compute total loss, backward, step, etc
        loss = class_loss + rot_loss + cent_loss
        loss.backward()
        optimizer.step()
        # by doing so, cl_lambda would not impact on the learning of centers
        if args.center_loss:
            for param in criterion_center.parameters():
                param.grad.data *= (1. / args.cl_lambda)
            optimizer_center.step()

        preds        = torch.argmax(obj_cls_output, dim=1)
        cls_correct += (preds == data_label).sum().item()

        preds        = torch.argmax(rot_cls_output, dim=1)
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
