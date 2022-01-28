import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler
from center_loss import CenterLoss

from tqdm import tqdm

#### Implement Step1

def _do_epoch(args, feature_extractor, rot_cls, obj_cls, get_rotation_classifiers, source_loader, optimizer, device, cls_criterion, rot_criterion):

    '''
    if args.center_loss:
        _obj_criterion = CenterLoss(num_classes=4, feat_dim=1024, use_gpu=torch.cuda.is_available)
        obj_criterion = lambda x, y: _obj_criterion(x, y) * args.cl_lambda;
    else:
        obj_criterion = nn.CrossEntropyLoss()
    '''

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
        if args.center_loss:  #version 2: we are using Discriminator 
            rot_cls_output,features = map(list,zip(*[classifiers[idx](output_rot_output_cat[idx]) for idx in it])) #version 2: features from first layer of R1
            #rot_cls_output,features = classifiers[0](output_rot_output_cat) #versione per single-head
            rot_cls_output = torch.vstack(rot_cls_output)
            features = torch.vstack(features)                   
        else: #otherwise we are using Classifier here (also version 1: features from feature extractor)
            rot_cls_output = torch.vstack([classifiers[idx](output_rot_output_cat[idx]) for idx in it])   

        class_loss  = cls_criterion(obj_cls_output, data_label)
        loss_ce, loss_cl    = rot_criterion(rot_cls_output, data_rot_label, output_rot_output_cat)
        loss        = class_loss + loss_ce + loss_cl

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

    return class_loss, acc_cls, loss_ce, loss_cl, acc_rot


def step1(args, feature_extractor, rot_cls, obj_cls, get_rotation_classifiers, source_loader, device):
    optimizer, scheduler = get_optim_and_scheduler(feature_extractor, rot_cls, obj_cls, args.epochs_step1, args.learning_rate, args.train_all, args.multihead, args.center_loss)

    cls_criterion = nn.CrossEntropyLoss()

    def _rot_criterion():
        rot_criterion_ce = nn.CrossEntropyLoss()
        if args.center_loss and args.cl_lambda > 0:
            #rot_criterion_cl = CenterLoss(num_classes=4, feat_dim=1024, use_gpu=torch.cuda.is_available())
            criterion_center = CenterLoss(num_classes=4, feat_dim=256, use_gpu=True, device=device) #version 2: features from first layer of R1
            #optimizer_center = torch.optim.SGD(criterion_center.parameters(), lr=args.learning_rate_center) #version a: used a specified LR for center loss

        a1, l = args.weight_RotTask_step1, args.cl_lambda
        if args.center_loss:
            optimizer.add_param_group({'params' : criterion_center.parameters()})
        def rot_criterion(scores, labels, feat_maps):
            # Does this have to use both, or... ?
            loss_ce = rot_criterion_ce(scores, labels) * a1
            loss_cl = criterion_center(feat_maps, labels) * l if args.center_loss else 0.0
            for param in loss_cl.parameters():
                param.grad.data *= (args.learning_rate_center / (args.cl_lambda * args.learning_rate))
            return loss_ce, loss_cl
        return rot_criterion

    rot_criterion = _rot_criterion()

    feature_extractor.train()
    obj_cls.train()

    if args.multihead:
        for head in rot_cls:
            head.train()
    else:
        rot_cls.train()

    for epoch in range(args.epochs_step1):
        print(f'Epoch {epoch+1}/{args.epochs_step1}')
        class_loss, acc_cls, loss_ce, loss_cl, acc_rot = _do_epoch(args, feature_extractor, rot_cls, obj_cls, get_rotation_classifiers, source_loader, optimizer, device, cls_criterion, rot_criterion)
        print(f"\tClass Loss    : {class_loss.item():.4f}")
        print(f"\tRot   Loss    : {loss_ce.item():.4f}")
        if args.center_loss:
            print(f"\tCenter   Loss    : {loss_cl.item():.4f}")
        print(f"\tClass Accuracy: {acc_cls*100:.2f}%")
        print(f"\tRot   Accuracy: {acc_rot*100:.2f}%")
        scheduler.step()
