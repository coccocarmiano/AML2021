import argparse
import copy

import torch
import data_helper
from resnet import resnet18_feat_extractor, Classifier, RotationDiscriminator
from optimizer_helper import get_optim_and_scheduler, get_optim_scheduler_loss_center_loss
from step1_KnownUnknownSep import step1
from eval_target import target_separation
from step2_SourceTargetAdapt import step2
from random import randint
import pickle
import os
from step1_KnownUnknownSep import plot_step1_accuracy_loss
from step2_SourceTargetAdapt import plot_step2_accuracy_loss, plot_eval_performance

SCRIPT_PATH = os.path.dirname(__file__)
MODEL_PATH = os.path.join(SCRIPT_PATH, "models")

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)


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
    parser.add_argument("--learning_rate_CL", type=float, default=0.5, help="Learning rate for center loss")

    parser.add_argument("--epochs_step1", type=int, default=10, help="Number of epochs of step1 for known/unknown separation")
    parser.add_argument("--epochs_step2", type=int, default=10, help="Number of epochs of step2 for source-target adaptation")

    parser.add_argument("--train_all", default=True, action="store_true", help="If true, all network weights will be trained")

    parser.add_argument("--weight_RotTask_step1", type=float, default=0.5, help="Weight for the rotation loss in step1")
    parser.add_argument("--weight_RotTask_step2", type=float, default=0.5, help="Weight for the rotation loss in step2")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for the known/unkown separation")
    parser.add_argument("--weight_CL", type=float, default=0.1, help="Weight for the center loss in step1")

    # tensorboard logger
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--folder_name", default=None, help="Used by the logger to save logs")
    parser.add_argument("--try_load", default=False, action="store_true", help="Try to load the models from a previous run if the configuration matches")

    # variants
    parser.add_argument("--multihead", default=False, action="store_true", help="If true will use multi-head rotation classifier")
    parser.add_argument("--center_loss", default=False, action="store_true", help="If true will use center loss")

    # specific runs
    parser.add_argument("--step1_only", default=False, action="store_true", help="Start the step1 only")
    parser.add_argument("--step2_only", default=False, action="store_true", help="Start the step2 only. It requires that try_load is set to true"
                                                                                 " and there must exist a previous run for this configuration")
    parser.add_argument("--eval_only", default=False, action="store_true", help="Start the final evaluation only. It requires that try_load is set to true"
                                                                                " and there must exist a previous run for this configuration")

    #return parser.parse_known_args()[0]
    return parser.parse_args()

class Trainer:
    def __init__(self, args):
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- MODELS ----- #
        # Step 1
        # E1: Feature extractor
        # C1: Object classifier
        # R1: Rotation classifier
        self.E1 = resnet18_feat_extractor().to(self.device)
        self.C1 = Classifier(512, self.args.n_classes_known + 1).to(self.device)
        n_heads = 1 if not args.multihead else args.n_classes_known
        self.R1 = RotationDiscriminator(input_size=1024, hidden_size=256, out_classes=4, n_heads=n_heads).to(self.device)

        # Optimizers and schedulers
        self.O1, self.scheduler1 = get_optim_and_scheduler(self.E1, self.C1, self.R1, args.epochs_step1, args.learning_rate, args.train_all)
        if args.center_loss:
            self.O1_CL, self.scheduler1_CL, self.criterion_CL = get_optim_scheduler_loss_center_loss(args.learning_rate_center, args.epochs_step1, self.device)
        else:
            self.O1_CL, self.scheduler1_CL, self.criterion_CL = None, None, None


        # History of the training for the step1
        self.history1 = {}
        self.history1['tot_loss'] = []
        self.history1['C_loss'] = []
        self.history1['R_loss'] = []
        self.history1['CL_loss'] = []
        self.history1['C_acc'] = []
        self.history1['R_acc'] = []

        # Separation performance
        self.auc = None
        self.separation_accuracy = None

        # Used for naming the known target dataset and the new source dataset after the separation
        self.rand = randint(0, 1e6)

        # Step 2
        # E2: Feature extractor: start from the trained E1 from the step 1
        # C2: Object classifier: start from the trained C1 from the step 1
        # R2: Rotation classifier: it starts as a new classifier (single-head)
        self.E2 = resnet18_feat_extractor().to(self.device)
        self.C2 = Classifier(512, self.args.n_classes_known + 1).to(self.device)
        self.R2 = RotationDiscriminator(input_size=1024, hidden_size=256, out_classes=4, n_heads=1).to(self.device)

        self.O2, self.scheduler2 = get_optim_and_scheduler(self.E2, self.C2, self.R2, args.epochs_step2, args.learning_rate, args.train_all)

        # History of the training for the step 2
        self.history2 = {}
        self.history2['tot_loss'] = []
        self.history2['C_loss'] = []
        self.history2['R_loss'] = []
        self.history2['C_acc'] = []
        self.history2['R_acc'] = []
        self.history2['eval_HOS'] = []
        self.history2['eval_OS'] = []
        self.history2['eval_UNK'] = []
        self.history2['eval_C_loss'] = []

        # Try load from file the models from a previous configuration
        self.loaded = False
        if args.try_load:
            self.loaded = self.try_load()
            if self.loaded:
                print(f"Model correctly loaded from file.")

        # -------------- #

        # ---- DATA LOADERS ---- #
        # Loading the source and target for the step1 (separation)
        # Source loader for the step 1 (known source in train mode)
        source_path_file = f"txt_list/{args.source}_known.txt" 
        self.source_loader = data_helper.get_train_dataloader(args, source_path_file)

        # This must remain untouched and it should be used only for the separation and for the final evaluation (whole target in eval mode)
        target_path_file = f"txt_list/{args.target}.txt"
        self.target_loader_eval = data_helper.get_val_dataloader(args, target_path_file)

        # -------------- #

        # PRINTING AND DEBUGGING
        print(f"Source known: {args.source} [{len(self.source_loader.dataset)} samples]")
        print(f"Target known+unknown: {args.target} [{len(self.target_loader_eval.dataset)} samples]")

        # Visualize some images from the Known Source and from the Target (known or unknown)
        data_helper.visualize_img(self.source_loader)
        data_helper.visualize_img(self.target_loader_eval)
        # ----------------- #


    def trainer_step1(self):
        print("Step One -- Training")
        hist1 = step1(self.args, self.E1, self.C1, self.R1, self.source_loader, self.device, self.O1, self.scheduler1,
                    optimizer_CL=self.O1_CL, scheduler_CL=self.scheduler1_CL, criterion_CL=self.criterion_CL)

        self.history1['tot_loss'].extend(hist1['tot_loss'])
        self.history1['C_loss'].extend(hist1['C_loss'])
        self.history1['R_loss'].extend(hist1['R_loss'])
        self.history1['CL_loss'].extend(hist1['CL_loss'])
        self.history1['C_acc'].extend(hist1['C_acc'])
        self.history1['R_acc'].extend(hist1['R_acc'])

    def trainer_target_separation(self):
        print("Target Known/Unknown Separation")
        self.auc, self.separation_accuracy = target_separation(self.args, self.E1, self.C1, self.R1, self.target_loader_eval, self.device, self.rand)

        print("Adding known source samples to the newly generated file... ", end="")
        filepath = f'new_txt_list/{self.args.source}_known_{str(self.rand)}.txt'
        with open(filepath, "a") as f:
            pairs = zip(self.source_loader.dataset.names, self.source_loader.dataset.labels)
            for (file, label) in pairs:
                f.write(f"{file} {str(label)}\n")
        print("Done.")

    def trainer_plot_step1(self):
        plot_step1_accuracy_loss(self.args, self.history1)

    def trainer_step2(self):
        # Before doing step2, deepcopying the E and C from step1
        if not self.loaded:
            self.E2 = copy.deepcopy(self.E1)
            self.C2 = copy.deepcopy(self.C1)

        # Build new dataloaders
        # New source (source + target unknown according to separation)
        source_path_file   = f"new_txt_list/{self.args.source}_known_{str(self.rand)}.txt"
        self.source_loader = data_helper.get_train_dataloader(self.args, source_path_file)

        # New target (target known according to separation)
        target_path_file = f"new_txt_list/{self.args.target}_known_{str(self.rand)}.txt"
        self.target_loader_train = data_helper.get_train_dataloader(self.args, target_path_file)

        print(f"Train Size for C2 (Source + Target Unknown): {len(self.source_loader.dataset)}")
        print(f"Train Size for R2 (Target known after separation): {len(self.target_loader_train.dataset)}")

        print("Step 2 -- Domain Adaptation")
        hist2 = step2(self.args, self.E2, self.C2, self.R2, self.source_loader, self.target_loader_train,
                                       self.target_loader_eval, self.device, self.O2, self.scheduler2)

        self.history2['tot_loss'].extend(hist2['tot_loss'])
        self.history2['C_loss'].extend(hist2['C_loss'])
        self.history2['R_loss'].extend(hist2['R_loss'])
        self.history2['C_acc'].extend(hist2['C_acc'])
        self.history2['R_acc'].extend(hist2['R_acc'])
        self.history2['eval_HOS'].extend(hist2['eval_HOS'])
        self.history2['eval_OS'].extend(hist2['eval_OS'])
        self.history2['eval_UNK'].extend(hist2['eval_UNK'])
        self.history2['eval_C_loss'].extend(hist2['eval_C_loss'])

    def trainer_plot_step2(self):
        plot_step2_accuracy_loss(self.args, self.history2)

    def trainer_plot_evaluation(self):
        plot_eval_performance(self.args, self.history2)

    # ------ SAVING AND LOADING ------- #
    def get_config_file_name(self):
        s = f"S-{self.args.source}-T-{self.args.target}-MH-{self.args.multihead}-CL-{self.args.center_loss}-"
        s += f"lr-{self.learning_rate:.5f}-"
        if self.args.center_loss:
            s += f"lr_cl-{self.learning_rate_center:.5f}-"
        s += f"A1-{self.args.weight_RotTask_step1}-A2-{self.args.weight_RotTask_step2}"
        if self.args.center_loss:
            s += f"-A_cl-{self.args.cl_lambda}"
        s += ".tar"
        return s

    def save(self):
        d = {
            'E1_state_dict': self.E1.state_dict(),
            'C1_state_dict': self.C1.state_dict(),
            'R1_state_dict': self.R1.state_dict(),
            'O1_state_dict': self.O1.state_dict(),
            'scheduler1_state_dict': self.scheduler1.state_dict(),
            'history1': self.history1,
            'auc': self.auc,
            'rand': self.rand,
            'separation_accuracy': self.separation_accuracy,
            'E2_state_dict': self.E2.state_dict(),
            'C2_state_dict': self.C2.state_dict(),
            'R2_state_dict': self.R2.state_dict(),
            'O2_state_dict': self.O2.state_dict(),
            'scheduler2_state_dict': self.scheduler2.state_dict(),
            'history2': self.history2
        }

        if self.args.center_loss:
            d['O1_CL_state_dict'] = self.O1_CL.state_dict()
            d['scheduler1_CL_state_dict'] = self.scheduler1_CL.state_dict()
            d['criterion_CL_state_dict'] = self.criterion_CL.state_dict()

        # Build file name according to the configuration
        f_name = self.get_config_file_name()
        path = os.path.join(MODEL_PATH, f_name)
        torch.save(d, path)

    def try_load(self):
        f_name = self.get_config_file_name()
        path = os.path.join(MODEL_PATH, f_name)
        if os.path.exists(path):
            d = torch.load(path)
            if 'E1_state_dict' not in d or 'C1_state_dict' not in d or 'R1_state_dict' not in d or \
                'O1_state_dict' not in d or 'scheduler1_state_dict' not in d or 'E2_state_dict' not in d or \
                'C2_state_dict' not in d or 'R2_state_dict' not in d or 'O2_state_dict' not in d or \
                'scheduler2_state_dict' not in d or 'rand' not in d:
                return False
            if self.args.center_loss:
                if 'O1_CL_state_dict' not in d or 'scheduler1_CL_state_dict' not in d or 'criterion_CL_state_dict' not in d:
                    return False

            self.E1.load_state_dict(d['E1_state_dict'])
            self.C1.load_state_dict(d['C1_state_dict'])
            self.R1.load_state_dict(d['R1_state_dict'])
            self.O1.load_state_dict(d['O1_state_dict'])
            self.scheduler1.load_state_dict(d['scheduler1_state_dict'])
            self.history1 = d['history1']
            self.auc = d['auc']
            self.rand = d['rand']
            self.separation_accuracy = d['separation_accuracy']
            self.E2.load_state_dict(d['E2_state_dict'])
            self.C2.load_state_dict(d['C2_state_dict'])
            self.R2.load_state_dict(d['R2_state_dict'])
            self.O2.load_state_dict(d['O2_state_dict'])
            self.scheduler2.load_state_dict(d['scheduler2_state_dict'])
            self.history2 = d['history2']

            if self.args.center_loss:
                self.O1_CL.load_state_dict(d['O1_CL_state_dict'])
                self.scheduler1_CL.load_state_dict(d['scheduler1_CL_state_dict'])
                self.criterion_CL.load_state_dict(d['criterion_CL_state_dict'])

            return True
        return False


def main():
    args = get_args()
    trainer = Trainer(args)

    # Do only specific steps
    if args.step1_only:
        trainer.trainer_step1()
        trainer.trainer_target_separation()
        return
    if args.step2_only:
        trainer.trainer_step2()
        trainer.trainer_final_eval()
        return
    if args.eval_only:
        trainer.trainer_final_eval()
        return

    # Do everything
    trainer.trainer_step1()
    trainer.trainer_target_separation()
    trainer.trainer_step2()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()