import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler

from itertools import cycle
from tqdm import tqdm
from center_loss import CenterLoss

def _do_epoch(args, feature_extractor, rot_cls, obj_cls, get_rotation_classifiers, source_loader, target_loader_train, target_loader_eval, optimizer, device):

    cls_criterion = nn.CrossEntropyLoss()
    rot_criterion = nn.CrossEntropyLoss()

    if args.center_loss:
        center_criterion = CenterLoss(num_classes=4, feat_dim=256, use_gpu=True, device=device)
        center_optimizer = torch.optim.SGD(center_criterion.parameters(), lr=args.learning_rate_center)


    target_loader_train = cycle(target_loader_train)
    cls_correct, rot_correct = 0, 0
    cls_tot,     rot_tot     = 0, 0

    for data_source, data_source_label, _, _ in tqdm(source_loader):
        optimizer.zero_grad()

        (data_target, _, data_target_rot, data_target_rot_label) = next(target_loader_train)

        data_source_label[data_source_label > 44] = 45

        # Move everything to the used device ( CPU/GPU )
        data_source           = data_source.to(device)
        data_target           = data_target.to(device)
        data_target_rot       = data_target_rot.to(device)
        data_source_label     = data_source_label.to(device)
        data_target_rot_label = data_target_rot_label.to(device)

        # Extract:
        # Class Label from Source Domain
        # Rotation Label from Target Domain
        feature_extractor_output            = feature_extractor(data_source)
        feature_extractor_output_target     = feature_extractor(data_target)
        feature_extractor_output_target_rot = feature_extractor(data_target_rot)

        obj_cls_output = obj_cls(feature_extractor_output)
        rot_cls_output = torch.cat((feature_extractor_output_target, feature_extractor_output_target_rot), dim=1)

        if args.center_loss:
            # Version 2: Using `Discriminator` classifier

            # First we get a `zip` of both outputs and featurs
            output_and_feats = [ rot_cls(sample) for sample in rot_cls_output ]

            # Then we unzip it into two tuples and covert them into two lists and then into tensors
            rot_cls_output, features = zip(*output_and_feats)

            rot_cls_output = list(rot_cls_output)
            features = list(features)

            features = torch.vstack(features)
            rot_cls_output = torch.vstack(rot_cls_output)

            # Single-head version (???)
            # rot_cls_output,features = classifiers[0](rot_cls_output)
        # otherwise we are using Classifier here (also version 1: features from feature extractor)
        else:
            rot_cls_output = torch.vstack([rot_cls(sample) for sample in rot_cls_output ])

        # Evaluate loss
        class_loss  = cls_criterion(obj_cls_output, data_source_label)
        rot_loss    = rot_criterion(rot_cls_output, data_target_rot_label) * args.weight_RotTask_step2
        cent_loss = .0

        if args.center_loss:
            cent_loss = center_criterion(features, data_rot_label) * args.cl_lambda

        loss = class_loss + rot_loss + cent_loss
        loss.backward()
        optimizer.step()

        # by doing so, cl_lambda would not impact on the learning of centers
        if args.center_loss:
            for param in center_criterion.parameters():
                param.grad.data *= (1. / args.cl_lambda)
            center_optimizer.step()

        cls_pred     = torch.argmax(obj_cls_output, dim=1)
        cls_correct += (cls_pred == data_source_label).sum().item()
        rot_pred     = torch.argmax(rot_cls_output, dim=1)
        rot_correct += (rot_pred == data_target_rot_label).sum().item()

        cls_tot += data_source_label.size(0)
        rot_tot += data_target_rot_label.size(0)

    acc_cls = cls_correct / cls_tot * 100
    acc_rot = rot_correct / rot_tot * 100

    print(f"\tClass    Loss: {class_loss.item():.4f}")
    print(f"\tRotation Loss: {rot_loss.item():.4f}")
    if args.center_loss:
        print(f"\tCenter Loss: {cent_loss.item():.4f}")
    print(f"\tRotation Acc.: {acc_rot:.2f}%")
    print(f"\tClass    Acc.: {acc_cls:.2f}%")


    feature_extractor.eval()
    obj_cls.eval()

    # kwn_acc   = OS*
    # unknw_acc = UNK
    tot_known, tot_unkwn, known_correct, unknw_correct  = 0, 0, 0, 0

    with torch.no_grad():
        for data, class_l, _, _ in tqdm(target_loader_eval):

            data, class_l = data.to(device), class_l.to(device)
            feature_extractor_output  = feature_extractor(data)
            obj_cls_output            = obj_cls(feature_extractor_output)
            cls_pred = torch.argmax(obj_cls_output, dim=1)


            known_mask    = class_l < 45
            unknw_mask    = class_l > 44

            class_l[unknw_mask] = 45

            tot_known    += known_mask.sum().item()
            tot_unkwn    += unknw_mask.sum().item()
            
            known_correct   += (cls_pred[known_mask] == class_l[known_mask]).sum().item()
            unknw_correct   += (cls_pred[unknw_mask] == class_l[unknw_mask]).sum().item()

        known_acc = known_correct / tot_known
        unknw_acc = unknw_correct / tot_unkwn

        hos = 2 / ( 1.0 / float(known_acc) + 1.0 / float(unknw_acc) )

    return known_acc, unknw_acc, hos


def step2(args, feature_extractor, rot_cls, obj_cls, get_rotation_classifiers, source_loader, target_loader_train, target_loader_eval, device):
    """
    Returns: Tuple(OS, UNK, HOS, OSD, RSD), Tuple is selected based on the highest scoring HOS
    OS = Known Accuracy
    UNK = Unknown Accuracy
    HOS = Harmonic Mean of OS and UNK
    OSD = Object Classifier State Dict
    RSD = Rotation Classifier State Dict
    """
    ## From "On the Effectivnes of ...", LR is doubled in this step
    ## TODO: IMPLEMENT DOUBLING OF LEARNING RATE
    optimizer, scheduler = get_optim_and_scheduler(feature_extractor, rot_cls, obj_cls, args.epochs_step2, args.learning_rate, args.train_all, False)
    best_values = (0, 0, 0, 0, 0)
    best = .0
    for epoch in range(args.epochs_step2):
        print(f"Epoch {epoch+1}/{args.epochs_step2}")
        known_acc, unknw_acc, hos = _do_epoch(args, feature_extractor, rot_cls, obj_cls, get_rotation_classifiers, source_loader, target_loader_train, target_loader_eval, optimizer, device)
        
        print(f"\tOS : {known_acc * 100:.2f}%")
        print(f"\tUNK: {unknw_acc * 100:.2f}%")
        print(f"\tHOS: {hos * 100:.2f}%")

        if hos > best:
            best = hos
            best_values = (known_acc, unknw_acc, hos, obj_cls, rot_cls)
        scheduler.step()
    
    return best_values
