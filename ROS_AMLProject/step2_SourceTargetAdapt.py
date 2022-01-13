
import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler
from itertools import cycle
import numpy as np

def _do_epoch(args, feature_extractor, rot_cls, obj_cls, get_rotation_classifiers, source_loader, target_loader_train, target_loader_eval, optimizer, device):

    if args.center_loss:
        raise Exception("Implement Center Loss")
    else:
        criterion = nn.CrossEntropyLoss()
    
    feature_extractor.train()
    obj_cls.train() # Should the classifier be re-initialized ?

    if args.multihead:
        for head in rot_cls:
            head.train()
    else:
        rot_cls.train()

    target_loader_train = cycle(target_loader_train)
    cls_correct, rot_correct = 0, 0
    cls_tot,     rot_tot     = 0, 0

    for it, (data_source, data_source_label, _, _) in enumerate(source_loader):
        optimizer.zero_grad()

        (data_target, _, data_target_rot, data_target_rotation_label) = next(target_loader_train)

        data_source_label[data_source_label > 44] = 45

        # Move everything to the used device ( CPU/GPU )
        data_source, data_source_label                            = data_source.to(device), data_source_label.to(device)
        data_target, data_target_rot, data_target_rotation_label  = data_target.to(device), data_target_rot.to(device), data_target_rotation_label.to(device)

        # Extract:
        # Class Label from Source Domain
        # Rotation Label from Target Domain
        feature_extractor_output            = feature_extractor(data_source)
        feature_extractor_output_target     = feature_extractor(data_target)
        feature_extractor_output_target_rot = feature_extractor(data_target_rot)

        obj_cls_output        = obj_cls(feature_extractor_output)
        output_rot_output_cat = torch.cat((feature_extractor_output_target, feature_extractor_output_target_rot), dim=1)

        classifiers    = get_rotation_classifiers(data_source_label)
        it             = range(len(classifiers))
        rot_cls_output = torch.vstack( [ classifiers[idx](output_rot_output_cat[idx]) for idx in it ] )

        # Evaluate loss
        class_loss  = criterion(obj_cls_output, data_source_label)
        rot_loss    = criterion(rot_cls_output, data_target_rotation_label) * args.weight_RotTask_step2
        loss        = class_loss + rot_loss

        loss.backward()
        optimizer.step()

        cls_pred     = torch.argmax(obj_cls_output, dim=1)
        cls_correct += (cls_pred == data_source_label).sum().item()
        rot_pred     = torch.argmax(rot_cls_output, dim=1)
        rot_correct += (rot_pred == data_target_rotation_label).sum().item()

        cls_tot += data_source_label.size(0)
        rot_tot += data_target_rotation_label.size(0)

    acc_cls = cls_correct / cls_tot * 100
    acc_rot = rot_correct / rot_tot * 100

    print(f"\tRotation Loss: {rot_loss.item():.4f}")
    print(f"\tClass    Loss: {class_loss.item():.4f}")
    print(f"\tRotation Acc.: {acc_rot:.2f}%")
    print(f"\tClass    Acc.: {acc_cls:.2f}%")


    feature_extractor.eval()
    obj_cls.eval()

    # kwn_acc   = OS*
    # unknw_acc = UNK
    tot_known, tot_unkwn, known_correct, unknw_correct  = 0, 0, 0, 0

    with torch.no_grad():
        for it, (data, class_l, _, _) in enumerate(target_loader_eval):

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
    ## IMPLEMENT DOUBLING OF LEARNING RATE
    optimizer, scheduler = get_optim_and_scheduler(feature_extractor, rot_cls, obj_cls, args.epochs_step2, args.learning_rate, args.train_all, args.multihead)
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
