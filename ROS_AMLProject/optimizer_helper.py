from torch import optim
from center_loss import CenterLoss
from torch.cuda import is_available as is_cuda_available

def get_optim_and_scheduler(feature_extractor,rot_cls,obj_cls, epochs, lr, train_all, multihead, center_loss):

    def chain(to_chain):
        t = []
        for l in to_chain:
            t.extend(l)
        return t

    params = list(obj_cls.parameters())
    params.extend(feature_extractor.parameters())
    
    if multihead:
        params.extend(chain( [ head.parameters() for head in rot_cls] ))
    else:
        params.extend(list(rot_cls.parameters()))

    optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, lr=lr)
    step_size = int(epochs * .8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)

    return optimizer, scheduler