
import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler
from itertools import cycle
import numpy as np

def _do_epoch(args,feature_extractor,rot_cls,obj_cls,source_loader,target_loader_train,target_loader_eval,optimizer,device):

    # Step2 
    # 
    # The unknw_accnown part of the target will be used to train the unknw_accnown class in Step2, the 
    # known part instead will be used for the source-target adaptation. 
    #       - From the assignment PDF

    criterion = nn.CrossEntropyLoss()
    feature_extractor.train()
    obj_cls.train() # Should the classifier be re-initialized ?
    rot_cls.train() # Should the classifier be re-initialized ?

    target_loader_train = cycle(target_loader_train)
    cls_correct, rot_correct, cls_tot, rot_tot = 0, 0, 0, 0

    for it, (data_source, data_source_label, _, _) in enumerate(source_loader):

        # Reset Gradients
        optimizer.zero_grad()

        # We're using two iterators at once, one for the source and one for the target.
        (data_target, _, data_target_rot, data_target_rotation_label) = next(target_loader_train)

        # UNDER TEST!
        # cnt = (data_source_label > 44).sum()
        # print(f"Found {cnt} samples with label > 44 (Unknown) in {len(data_source_label)} samples")
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

        obj_cls_output  = obj_cls(feature_extractor_output)
        temp            = torch.cat((feature_extractor_output_target, feature_extractor_output_target_rot), dim=1)
        rot_cls_output  = rot_cls(temp) 

        # Evaluate loss
        class_loss  = criterion(obj_cls_output, data_source_label)
        rot_loss    = criterion(rot_cls_output, data_target_rotation_label)
        loss        = class_loss + args.weight_RotTask_step2 * rot_loss

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

    print("Class Loss %.4f, Class Accuracy %.4f,Rot Loss %.4f, Rot Accuracy %.4f" % (class_loss.item(), acc_cls, rot_loss.item(), acc_rot))


    #### Implement the final evaluation step, computing OS*, UNK and HOS
    feature_extractor.eval()
    obj_cls.eval()
    rot_cls.eval()

    # kwn_acc   = OS*
    # unknw_acc = UNK
    tot_known, tot_unknw, known_correct, unknw_correct  = 0, 0, 0, 0

    with torch.no_grad():
        for it, (data, class_l, _, _) in enumerate(target_loader_eval):
            data, class_l = data.to(device), class_l.to(device)
            feature_extractor_output  = feature_extractor(data)
            obj_cls_output            = obj_cls(feature_extractor_output)

            cls_pred = torch.argmax(obj_cls_output, dim=1)

            # Are this masks correct? Need to check
            class_l[class_l > 44] = 45
            tot_known    += class_l[class_l < 45].sum()
            tot_unkwn    += class_l[class_l > 44].sum()
            
            known_correct   += (class_l == cls_pred)[class_l < 45].sum()
            unknw_correct   += ((class_l > 44) * (cls_pred > 44)).sum()
            
        
        
        known_acc = float(known_correct) / float(tot_known)
        unknw_acc = float(unknw_correct) / float(tot_unknw)

        if known_acc < 0.001:
            known_acc = 0.001
        
        if unknw_acc < 0.001:
            unknw_acc = 0.001

        print(known_acc, unknw_acc)

        hos = 2 / ( 1.0 / float(known_acc) + 1.0 / float(unknw_acc) )

        print(f'\nkwn_acc={known_acc * 100:.2f}, unknw_acc={unknw_acc * 100 : 2f}, HOS={hos * 100 : .2f}')



def step2(args,feature_extractor,rot_cls,obj_cls,source_loader,target_loader_train,target_loader_eval,device):
    optimizer, scheduler = get_optim_and_scheduler(feature_extractor,rot_cls,obj_cls, args.epochs_step2, args.learning_rate, args.train_all)
    for epoch in range(args.epochs_step2):
        _do_epoch(args,feature_extractor,rot_cls,obj_cls,source_loader,target_loader_train,target_loader_eval,optimizer,device)
        scheduler.step()
