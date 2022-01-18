import argparse

import torch

import data_helper
from resnet import resnet18_feat_extractor, Classifier

from step1_KnownUnknownSep import step1
from eval_target import evaluation
from step2_SourceTargetAdapt import step2

import pickle

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch training", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--source", default='Art', help="Source name")
    parser.add_argument("--target", default='Clipart', help="Target name")
    parser.add_argument("--n_classes_known", type=int, default=45, help="Number of known classes")
    parser.add_argument("--n_classes_tot", type=int, default=65, help="Number of unknown classes")

    # dataset path
    parser.add_argument("--path_dataset", default="./data", help="Path where the Office-Home dataset is located")

    # data augmentation
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--random_grayscale", default=0.1, type=float, help="Randomly greyscale the image")

    # training parameters
    parser.add_argument("--image_size", type=int, default=222, help="Image size")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")

    parser.add_argument("--epochs_step1", type=int, default=10, help="Number of epochs of step1 for known/unknown separation")
    parser.add_argument("--epochs_step2", type=int, default=10, help="Number of epochs of step2 for source-target adaptation")

    parser.add_argument("--train_all", type=bool, default=True, help="If true, all network weights will be trained")

    parser.add_argument("--weight_RotTask_step1", type=float, default=0.5, help="Weight for the rotation loss in step1")
    parser.add_argument("--weight_RotTask_step2", type=float, default=0.5, help="Weight for the rotation loss in step2")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for the known/unkown separation")

    # tensorboard logger
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--folder_name", default=None, help="Used by the logger to save logs")

    # variants
    parser.add_argument("--multihead", type=bool, default=False, help="If true will use multi-head rotation classifier")
    parser.add_argument("--center_loss", type=bool, default=False, help="If true will use center loss") # To be implemented
    return parser.parse_known_args()[0]
    #return parser.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialize the network with a number of classes equals to the number of known classes + 1 (the unknown class, trained only in step2)
        self.feature_extractor = resnet18_feat_extractor()
        self.obj_classifier = Classifier(512, self.args.n_classes_known+1).to(self.device)

        if args.multihead:
            self.rot_classifier = [ Classifier(1024, 4).to(self.device) for _ in range(args.n_classes_known+1) ]
        else:
            self.rot_classifier = Classifier(1024, 4).to(self.device)

        self.feature_extractor = self.feature_extractor.to(self.device)
        self.obj_cls = self.obj_classifier
        self.rot_cls = self.rot_classifier

        source_path_file = f"txt_list/{args.source}_known.txt" 
        self.source_loader = data_helper.get_train_dataloader(args, source_path_file)

        target_path_file = f"txt_list/{args.target}.txt"
        self.target_loader_train = data_helper.get_val_dataloader(args, target_path_file)
        self.target_loader_eval = data_helper.get_val_dataloader(args, target_path_file)

        print(f"Source known: {args.source} [{len(self.source_loader.dataset)}]")

        ### DEBUG andrea
        data_helper.visualize_img(self.source_loader) #batch of 5 images
        ### DEBUG andrea


        print(f"Target known+unknown: {args.target} [{len(self.target_loader_train.dataset)}]")

        ### DEBUG
        data_helper.visualize_img(self.target_loader_train) #just one image
        ### DEBUG

    def get_rotation_classifiers(self):
        # Wrapper Method
        multihead      = self.args.multihead
        rot_classifier = self.rot_classifier
        def _get_rotation_classifiers(class_labels):
            """
            Returns a Tuple with:
            a ) the list of classifiers to use, with `list[i]` being the classifier for class `class_labels[i]`
            b ) the `zip` of (`label`, `idx` ) for an easier iteration
            """
            n_samples = len(class_labels)
            if multihead:
                classifiers = [ self.rot_classifier[i] for i in class_labels ]
            else:
                classifiers = [ self.rot_classifier    for i in class_labels ]
            return classifiers

        return _get_rotation_classifiers
        
    def trainer_step1(self):
        print("Step One -- Training")
        step1(self.args, self.feature_extractor, self.rot_cls, self.obj_cls, self.get_rotation_classifiers(), self.source_loader, self.device)

        ### For Debug Purposes
        with open("obj-s1.pickle", "wb") as f:
            pickle.dump(self.obj_cls, f)
        with open("rot-s1.pickle", "wb") as f:
            pickle.dump(self.rot_cls, f)
        ### For Debug Purposes


    def trainer_evaluation(self):
        print("Evaluation -- Known/Unknown Separation")
        rand = evaluation(self.args, self.feature_extractor, 
                          self.rot_cls, self.obj_cls, self.get_rotation_classifiers(), self.target_loader_eval, self.device)

        print(f"Random: {rand}")

        ### For Debug Purposes
        with open('lastrand', 'w') as f:
            f.write(str(rand))
        with open("obj-ev.pickle", "wb") as f:
            pickle.dump(self.obj_cls, f)
        with open("rot-ev.pickle", "wb") as f:
            pickle.dump(self.rot_cls, f)
        ### For Debug Purposes
        

        print("Adding source samples to known target samples... ", end="")

        filepath = f'new_txt_list/{self.args.source}_known_{str(rand)}.txt'

        with open(filepath, "a") as f:
            pairs = zip(self.source_loader.dataset.names, self.source_loader.dataset.labels)
            for (file, label) in pairs:
                f.write(f"{file} {str(label)}\n")

        print("Done.")
        return rand

    def trainer_step2(self):
        ### For Debug Purposes
        with open("lastrand", "r") as lastrand:
            rand = int(lastrand.read())
        print(f"Random: {rand}")
        ### For Debug Purposes
        
        # new dataloaders
        source_path_file   = f"new_txt_list/{self.args.source}_known_{str(rand)}.txt"
        self.source_loader = data_helper.get_train_dataloader(self.args, source_path_file)

        target_path_file = f"new_txt_list/{self.args.target}_known_{str(rand)}.txt"
        self.target_loader_train = data_helper.get_train_dataloader(self.args, target_path_file)
        self.target_loader_eval  = data_helper.get_val_dataloader(self.args, target_path_file)

        print(f"Source Size (S+UNK): {len(self.source_loader.dataset)}")
        print(f"Target Size (TRAIN): {len(self.target_loader_train.dataset)}")
        print(f"Target Size (TEST ): {len(self.target_loader_eval.dataset)}")

        print("Step 2 -- Domain Adaptation")
        os, unk, hos, osd, rsd = step2(self.args, self.feature_extractor, self.rot_cls, self.obj_cls, self.get_rotation_classifiers(),
                                       self.source_loader, self.target_loader_train, self.target_loader_eval, self.device)
        print("Saving best performing model based on HOS")
        
        # These should actually be .pth models
        ### For Debug Purposes
        with open("obj-s2.pickle", "wb") as f:
            pickle.dump(osd, f)
        with open("rot-s2.pickle", "wb") as f:
            pickle.dump(rsd, f)
        ### For Debug Purposes
        
    def do_training(self):
        #self.trainer_step1()
        self.trainer_evaluation()
        self.traner_step2()

def main():
    args = get_args()
    trainer = Trainer(args)
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()