import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler

from tqdm import tqdm

#### Implement Step1

def _do_epoch(args, feature_extractor, rot_cls, obj_cls, get_rotation_classifiers, source_loader, optimizer, device):

    if args.center_loss:
        raise Exception("Implement Center Loss")
    else:
        criterion = nn.CrossEntropyLoss()

    feature_extractor.train()
    obj_cls.train()

    if args.multihead:
        for head in rot_cls:
            head.train()
    else:
        rot_cls.train()

    cls_correct, rot_correct, cls_tot, rot_tot = 0, 0, 0, 0

    for data, data_label, data_rot, data_rot_label in tqdm(source_loader):
        optimizer.zero_grad()

        data    , data_label     = data.to(device)    , data_label.to(device),
        data_rot, data_rot_label = data_rot.to(device), data_rot_label.to(device)
        
        feature_extractor_output     = feature_extractor(data)
        feature_extractor_output_rot = feature_extractor(data_rot)

        obj_cls_output        = obj_cls(feature_extractor_output)
        output_rot_output_cat = torch.cat((feature_extractor_output, feature_extractor_output_rot), dim=1)

        classifiers    = get_rotation_classifiers(data_label)
        it             = range(len(classifiers))
        rot_cls_output = torch.vstack([ classifiers[idx](output_rot_output_cat[idx]) for idx in it])

        class_loss  = criterion(obj_cls_output, data_label)
        rot_loss    = criterion(rot_cls_output, data_rot_label) * args.weight_RotTask_step1
        loss        = class_loss + rot_loss

        loss.backward()
        optimizer.step()
        
        preds        = torch.argmax(obj_cls_output, dim=1)
        cls_correct += (preds == data_label).sum().item()

        preds        = torch.argmax(rot_cls_output, dim=1)
        rot_correct += (preds == data_rot_label).sum().item()
        cls_tot     += data_label.size(0)
        rot_tot     += data_rot_label.size(0)


    acc_cls = cls_correct / cls_tot
    acc_rot = rot_correct / rot_tot

    return class_loss, acc_cls, rot_loss, acc_rot


def step1(args, feature_extractor, rot_cls, obj_cls, get_rotation_classifiers, source_loader, device):
    optimizer, scheduler = get_optim_and_scheduler(feature_extractor, rot_cls, obj_cls, args.epochs_step1, args.learning_rate, args.train_all, args.multihead)

    for epoch in range(args.epochs_step1):
        print(f'Epoch {epoch+1}/{args.epochs_step1}')
        class_loss, acc_cls, rot_loss, acc_rot = _do_epoch(args, feature_extractor, rot_cls, obj_cls, get_rotation_classifiers, source_loader, optimizer, device)
        print(f"\tClass Loss    : {class_loss.item():.4f}")
        print(f"\tRot   Loss    : {rot_loss.item():.4f}")
        print(f"\tClass Accuracy: {acc_cls*100:.2f}%")
        print(f"\tRot   Accuracy: {acc_rot*100:.2f}%")
        scheduler.step()
