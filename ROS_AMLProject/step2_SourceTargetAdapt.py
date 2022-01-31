import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler

from itertools import cycle
from tqdm import tqdm
from center_loss import CenterLoss

def _do_epoch(args, feature_extractor, rot_cls, obj_cls, source_loader, target_loader_train, target_loader_eval, optimizer, device):

    # Set training on
    rot_cls.train()
    obj_cls.train()
    feature_extractor.train()

    # Initialize a weight list
    # By setting the last class weight to twice the other
    # We achieve LR doubiling for the last class
    w = [ 1.0 for _ in range(args.n_classes_known + 1) ]
    w[-1] = 2.0
    w = torch.tensor(w).to(device)

    # Tensor type is automatically inferred. Should also take a `reduction` argument
    # THEORETICALLY we should leave it as is, it means `none` thus no normalization
    cls_criterion = nn.CrossEntropyLoss(weight=w)
    rot_criterion = nn.CrossEntropyLoss()

    if args.center_loss:
        center_criterion = CenterLoss(num_classes=4, feat_dim=256, use_gpu=True, device=device)
        center_optimizer = torch.optim.SGD(center_criterion.parameters(), lr=args.learning_rate_center)

    # We iterate over two datasets at once
    target_loader_train = cycle(target_loader_train)
    cls_correct, rot_correct = 0, 0
    cls_tot,     rot_tot     = 0, 0

    for data_source, data_source_label, _, _ in tqdm(source_loader):
        (data_target, _, data_target_rot, data_target_rot_label) = next(target_loader_train)

        # Zero-out the gradients
        optimizer.zero_grad()
        if args.center_loss:
            center_optimizer.zero_grad()

        # Move everything to the used device ( CPU/GPU )
        data_source           = data_source.to(device)
        data_target           = data_target.to(device)
        data_target_rot       = data_target_rot.to(device)
        data_source_label     = data_source_label.to(device)
        data_target_rot_label = data_target_rot_label.to(device)

        # 1. Extract features from E
        # Extracting source image features from E
        feature_extractor_output_source     = feature_extractor(data_source)
        # Extracting original target image features from E
        feature_extractor_output_target     = feature_extractor(data_target)
        # Extracting rotated target image features from E
        feature_extractor_output_target_rot = feature_extractor(data_target_rot)
        # Concatenate original+rotate target features
        feature_extractor_output_target_conc = torch.cat((feature_extractor_output_target, feature_extractor_output_target_rot), dim=1)

        # 2. Get the scores
        # For the source image, we get the output scores from C2
        obj_cls_target_scores = obj_cls(feature_extractor_output_source)

        # For the target image, we want the scores from R2
        # For the center loss version, we need to have both the features coming from the first layer of the discriminator
        # and the output scores coming out from the discriminator
        if args.center_loss:
            # 1. For each sample in the batch, get the discriminator features and output scores
            discriminator_features = []
            discriminator_scores = []
            for sample in feature_extractor_output_target_conc:
                discriminator_sample_scores, discriminator_sample_features = rot_cls(sample)
                discriminator_features.append(discriminator_sample_features)
                discriminator_scores.append(discriminator_sample_scores)

            # 2. Transform them back again into tensors (#batch_size, #features_dim)
            discriminator_features = torch.vstack(discriminator_features)
            discriminator_scores = torch.vstack(discriminator_scores)
        else:
            discriminator_scores = torch.vstack([rot_cls(sample)[0] for sample in feature_extractor_output_target_conc])

        # Evaluate losses
        class_loss  = cls_criterion(obj_cls_target_scores, data_source_label)
        rot_loss    = rot_criterion(discriminator_scores, data_target_rot_label) * args.weight_RotTask_step2
        cent_loss   = .0

        if args.center_loss:
            cent_loss = center_criterion(discriminator_features, data_target_rot_label) * args.cl_lambda

        loss = class_loss + rot_loss + cent_loss
        loss.backward()
        optimizer.step()

        # by doing so, cl_lambda would not impact on the learning of centers
        if args.center_loss:
            for param in center_criterion.parameters():
                param.grad.data *= (1. / args.cl_lambda)
            center_optimizer.step()

        cls_pred     = torch.argmax(obj_cls_target_scores, dim=1)
        cls_correct += (cls_pred == data_source_label).sum().item()
        rot_pred     = torch.argmax(discriminator_scores, dim=1)
        rot_correct += (rot_pred == data_target_rot_label).sum().item()

        cls_tot += data_source_label.size(0)
        rot_tot += data_target_rot_label.size(0)

    acc_cls = cls_correct / cls_tot * 100
    acc_rot = rot_correct / rot_tot * 100

    print("Train Dataset Stats:")
    print(f"\tClass  Loss: {class_loss.item():.4f}")
    print(f"\tRot    Loss: {rot_loss.item():.4f}")
    if args.center_loss:
        print(f"\tCenter Loss: {cent_loss.item():.4f}")
    print(f"\tRot    Acc.: {acc_rot:.2f}%")
    print(f"\tClass  Acc.: {acc_cls:.2f}%")


    # Final evaluation on the target using only C2
    # Disable training
    feature_extractor.eval()
    obj_cls.eval()
    rot_cls.eval()

    # kwn_acc   = OS*
    # unknw_acc = UNK
    tot_known, tot_unkwn, known_correct, unknw_correct  = 0, 0, 0, 0

    with torch.no_grad():
        for data, class_l, _, _ in tqdm(target_loader_eval):

            data, class_l            = data.to(device), class_l.to(device)
            feature_extractor_output = feature_extractor(data)
            obj_cls_target_scores           = obj_cls(feature_extractor_output)
            cls_pred                 = torch.argmax(obj_cls_target_scores, dim=1)


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


def step2(args, feature_extractor, rot_cls, obj_cls, source_loader, target_loader_train, target_loader_eval, device):
    """
    Returns: Tuple(OS, UNK, HOS, OSD, RSD), Tuple is selected based on the highest scoring HOS
    OS = Known Accuracy
    UNK = Unknown Accuracy
    HOS = Harmonic Mean of OS and UNK
    OSD = Object Classifier State Dict
    RSD = Rotation Classifier State Dict
    """
    optimizer, scheduler = get_optim_and_scheduler(feature_extractor, rot_cls, obj_cls, args.epochs_step2, args.learning_rate, args.train_all, False)
    best_values = (0, 0, 0, 0, 0)
    best = .0
    for epoch in range(args.epochs_step2):
        print(f"Epoch {epoch+1}/{args.epochs_step2}")
        known_acc, unknw_acc, hos = _do_epoch(args, feature_extractor, rot_cls, obj_cls, source_loader, target_loader_train, target_loader_eval, optimizer, device)
        
        print("Test Stats")
        print(f"\tOS : {known_acc * 100:.2f}%")
        print(f"\tUNK: {unknw_acc * 100:.2f}%")
        print(f"\tHOS: {hos * 100:.2f}%")

        if hos > best:
            best = hos
            best_values = (known_acc, unknw_acc, hos, obj_cls, rot_cls)
        scheduler.step()
    
    print(f"Last HOS: {hos*100:.2f}\nBest HOS {best*100:.2f}")
    return best_values
