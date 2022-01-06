
import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler

#### Implement Step1

def _do_epoch(args,feature_extractor,rot_cls,obj_cls,source_loader,optimizer,device):

    criterion = nn.CrossEntropyLoss()
    feature_extractor.train()
    obj_cls.train()
    rot_cls.train()

    cls_correct, rot_correct, cls_tot, rot_tot = 0, 0, 0, 0

    for it, (data, class_l, data_rot, rot_l) in enumerate(source_loader):
        data, class_l, data_rot, rot_l  = data.to(device), class_l.to(device), data_rot.to(device), rot_l.to(device)

        optimizer.zero_grad()
        feature_extractor_output     = feature_extractor(data)
        feature_extractor_output_rot = feature_extractor(data_rot)

        obj_cls_output = obj_cls(feature_extractor_output)
        u = torch.cat((feature_extractor_output, feature_extractor_output_rot), dim=1)
        rot_cls_output = rot_cls(u) 

        class_loss  = criterion(obj_cls_output, class_l)
        rot_loss    = criterion(rot_cls_output, rot_l)
        loss        = class_loss + args.weight_RotTask_step1 * rot_loss

        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(obj_cls_output, dim=1)
        cls_correct += (preds == class_l).sum().item()

        preds = torch.argmax(rot_cls_output, dim=1)
        rot_correct += (preds == rot_l).sum().item()
        cls_tot += class_l.size(0)
        rot_tot += rot_l.size(0)


    acc_cls = cls_correct / cls_tot * 100
    acc_rot = rot_correct / rot_tot * 100

    return class_loss, acc_cls, rot_loss, acc_rot


def step1(args,feature_extractor,rot_cls,obj_cls,source_loader,device):
    optimizer, scheduler = get_optim_and_scheduler(feature_extractor,rot_cls,obj_cls, args.epochs_step1, args.learning_rate, args.train_all)


    for epoch in range(args.epochs_step1):
        print('Epoch: ',epoch)
        class_loss, acc_cls, rot_loss, acc_rot = _do_epoch(args,feature_extractor,rot_cls,obj_cls,source_loader,optimizer,device)
        print("Class Loss %.2f, Class Accuracy %.2f,Rot Loss %.2f, Rot Accuracy %.2f" % (class_loss.item(),acc_cls,rot_loss.item(), acc_rot))
        scheduler.step()
