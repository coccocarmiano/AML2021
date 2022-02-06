from torch import optim
from center_loss import CenterLoss


def get_optim_and_scheduler(E, C, R, epochs, lr, train_all):
    R_params = list(R.parameters())

    if train_all:
        E_C_params = list(C.parameters()) + list(E.parameters())
        R_params = list(R.parameters())
        optimizer = optim.SGD([
            {'params': E_C_params, 'lr': lr/10},
            {'params': R_params}
        ], weight_decay=.0005, momentum=.9, lr=lr)
    else:
        params = list(C.parameters()) + R_params
        optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, lr=lr)

    step_size = int(epochs * .8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)

    return optimizer, scheduler

def get_optim_scheduler_loss_center_loss(lr, epochs, device):
    criterion_center = CenterLoss(num_classes=4, feat_dim=256, use_gpu=True, device=device)
    optimizer_center = optim.SGD(criterion_center.parameters(), weight_decay=.0005, momentum=.9, lr=lr)
    step_size = int(epochs * 10)
    scheduler = optim.lr_scheduler.StepLR(optimizer_center, step_size=step_size)

    return optimizer_center, scheduler, criterion_center
