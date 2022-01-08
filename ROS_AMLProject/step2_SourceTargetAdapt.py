
import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler
from itertools import cycle
import numpy as np


#### Implement Step2

def _do_epoch(args,feature_extractor,rot_cls,obj_cls,source_loader,target_loader_train,target_loader_eval,optimizer,device):

    criterion = nn.CrossEntropyLoss()
    feature_extractor.train()
    obj_cls.train()
    rot_cls.train()

    target_loader_train = cycle(target_loader_train)
    cls_correct, rot_correct, cls_tot, rot_tot = 0, 0, 0, 0

    for it, (data_source, class_l_source, _, _) in enumerate(source_loader):

        (data_target, _, data_target_rot, rot_l_target) = next(target_loader_train)

        data_source, class_l_source  = data_source.to(device), class_l_source.to(device)
        data_target, data_target_rot, rot_l_target  = data_target.to(device), data_target_rot.to(device), rot_l_target.to(device)

        optimizer.zero_grad()

        feature_extractor_output     = feature_extractor(data_source)
        feature_extractor_output_target = feature_extractor(data_target)
        feature_extractor_output_target_rot = feature_extractor(data_target_rot)

        obj_cls_output = obj_cls(feature_extractor_output)
        temp = torch.cat((feature_extractor_output_target, feature_extractor_output_target_rot), dim=1)
        rot_cls_output = rot_cls(temp) 

        class_loss  = criterion(obj_cls_output, class_l_source)
        rot_loss    = criterion(rot_cls_output, rot_l_target)
        loss = class_loss + args.weight_RotTask_step2*rot_loss

        loss.backward()

        optimizer.step()

        cls_pred = torch.argmax(obj_cls_output, dim=1)
        cls_correct += (cls_pred == class_l_source).sum().item()
        rot_pred = torch.argmax(rot_cls_output, dim=1)
        rot_correct += (rot_pred == rot_l_target).sum().item()

        cls_tot += class_l_source.size(0)
        rot_tot += rot_l_target.size(0)

    acc_cls = cls_correct / cls_tot * 100
    acc_rot = rot_correct / rot_tot * 100

    print("Class Loss %.4f, Class Accuracy %.4f,Rot Loss %.4f, Rot Accuracy %.4f" % (class_loss.item(), acc_cls, rot_loss.item(), acc_rot))


    #### Implement the final evaluation step, computing OS*, UNK and HOS
    feature_extractor.eval()
    obj_cls.eval()
    # rot_cls.eval()

    tot_known, tot_unk, OS_, UNK = 0, 0, 0, 0

    with torch.no_grad():
        for it, (data, class_l,_,_) in enumerate(target_loader_eval):
            feature_extractor_output  = feature_extractor(data)
            obj_cls_output = obj_cls(feature_extractor_output)

            cls_pred = torch.argmax(obj_cls_output, dim=1)

            # Are this masks correct? Need to check
            mask_known = (class_l != args.n_classes_known+1)
            mask_unk = (class_l == args.n_classes_known+1)

            tot_known = mask_known.sum()
            tot_unk = mask_unk.sum()

            OS_ += (cls_pred == class_l[mask_known]).sum().item()
            UNK += (cls_pred == class_l[mask_unk]).sum().item()
            # HOS = ???
        
        OS_ = OS_ / tot_known * 100
        UNK = UNK / tot_unk * 100

        print(f'\nOS_={OS_}, UNK={UNK}, HOS={...}')




def step2(args,feature_extractor,rot_cls,obj_cls,source_loader,target_loader_train,target_loader_eval,device):
    print("Help")
    pass
#     optimizer, scheduler = get_optim_and_scheduler(feature_extractor,rot_cls,obj_cls, args.epochs_step2, args.learning_rate, args.train_all)
# 
# 
#     for epoch in range(args.epochs_step2):
#         _do_epoch(args,feature_extractor,rot_cls,obj_cls,source_loader,target_loader_train,target_loader_eval,optimizer,device)
#         scheduler.step()
