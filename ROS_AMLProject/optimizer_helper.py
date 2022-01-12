from torch import optim


def get_optim_and_scheduler(feature_extractor,rot_cls,obj_cls, epochs, lr, train_all, multihead):

    def chain(list_of_lists):
        t = []
        for l in list_of_lists:
            t.extend(l)
        return t

    if multihead:
        rot_cls_params = chain( [ list(head.parameters()) for head in rot_cls ] )
    else:
        rot_cls_params = rot_cls.parameters()

    if train_all:
        params = list(rot_cls.parameters()) + rot_cls_params + list(feature_extractor.parameters())
    else:
        params = list(rot_cls.parameters()) + rot_cls_params

    optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, lr=lr)
    step_size = int(epochs * .8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)

    return optimizer, scheduler