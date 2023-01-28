import os
import argparse
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.model import Model
from src.dataset import DataModule
from src.utils import *
from src.augment import *

pl.seed_everything(123)

def train(args, model, dataset):
    checkpoint_callback = ModelCheckpoint(monitor='valid_f1_score',
                                          save_top_k=1,
                                          save_weights_only=True,
                                          filename='best-{epoch:02d}-{valid_avg_loss:.5f}-{valid_f1_score:.5f}',
                                          verbose=False,
                                          mode='max')
    trainer = pl.Trainer(
        accelerator = "gpu",
        devices = [args.gpu],
        max_epochs = args.epoch,
        precision = 16,
        fast_dev_run = False,
        enable_progress_bar = True,
        callbacks = checkpoint_callback,
        logger = CSVLogger(save_dir="./logs/" + args.dataset_name + "/", name=args.model + "_" + str(args.batch_size)),
    )
    time_start = datetime.now()
    trainer.fit(model, dataset)
    print("\nTraining execution time is: ", (datetime.now() - time_start))
    return

def main(args):
    data_path = os.path.join(args.dataset_path, args.dataset_name,'set.npz')
    model = Model(
        arch=args.model, 
        in_channels=1, 
        out_channels=1, 
        lr=args.learning_rate)
    dataset = DataModule(
        data_dir = data_path, 
        batch_size = args.batch_size,
        numworkers = args.numworkers,
        data_name = args.dataset_name,
        is_patch = args.is_patch,
        transform = get_train_augmentation)
    train(args, model, dataset)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model',default="UNet", type=str,
                        help='the name of model',
                        choices=['FR_UNet', 'R2AttU_Net', 'SegNet', 'NestedUNet', 'UNet_3Plus', 'AttU_Net', 'UNet'],
                        required=True)
    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('-p', '--is_patch', action="store_true",
                        help='patch image of slide window for whole image')
    parser.add_argument('-n', '--numworkers', default=0, type=int,
                        help='number of workers')
    parser.add_argument('-g', '--gpu', default=0)
    parser.add_argument('-e', '--epoch', default=50, type=int,
                        help='training epoch')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument('-dp', '--dataset_path', default="datasets", type=str,
                        help='the path of dataset')
    parser.add_argument('-dn', '--dataset_name', default="DRIVE", type=str,
                        help='the name of dataset',choices=['CHASEDB1','STARE','CHUAC','DCA1','STARE_u'],required=True)
    args = parser.parse_args()
    
    main(args)
    
    ## command -- python train.py -m UNet -b 512 -p -dn CHUAC