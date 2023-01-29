import os
import ttach as tta
from tqdm import tqdm
from monai.metrics import ConfusionMatrixMetric, DiceMetric
import argparse
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.model import Model
from src.dataset import *
from src.utils import *
from src.augment import *

pl.seed_everything(123)

mode_states = None

def evaluate_data(args, data_dir, batch_size, rel=None):
    h, w = get_h_w(args.dataset_name)
    val_aug = get_transform(rand_augment=None, stage='valid', height=h, width=w)  
    with np.load(data_dir, allow_pickle=True) as f:
        x_train_u, x_train_u_id = f['image'], f['image_name']
    if rel:
        rel_x_train_u, rel_x_train_u_id = [], []
        for reliable in rel:
            for idx, id in enumerate(x_train_u_id):
                if reliable[0] == id:
                    rel_x_train_u.append(x_train_u[idx])
                    rel_x_train_u_id.append(x_train_u_id[idx])
        del x_train_u, x_train_u_id
        x_train_u, x_train_u_id = np.array(rel_x_train_u), np.array(rel_x_train_u_id)
    unlabeled_dataset = ImageDataset(x_train_u, None, val_aug)
    return DataLoader(
        unlabeled_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.numworkers,
        pin_memory=True,
        drop_last=False), x_train_u_id

def evaluate(args, model, data, save_path, mode, name=None):
    if mode == 'pseudo':
        model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean').cuda(device=args.gpu)
    else: ## val
        model = model.cuda(device=args.gpu)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    confusion_metric = ConfusionMatrixMetric(include_background=False,
                                             metric_name=['f1 score', 'precision', 'recall'],
                                             reduction="mean", get_not_nans=False, compute_sample=True)
    model.eval()
    metrics = []
    img_t, gt_t = torch.Tensor(), torch.Tensor()
    pred_t = torch.Tensor()
    for id, (image, label) in enumerate(tqdm(data)):
        image = image.cuda(device=args.gpu)
        with torch.no_grad():
            logits = model(image)
        pr_mask = logits.sigmoid()
        pred_mask = (pr_mask > 0.5).float()

        if mode == 'val':
            label = label.cuda(device=args.gpu)
            dice_metric(pred_mask, label)
            confusion_metric(pred_mask, label)
            metrics.append(confusion_metric.aggregate())

        pred_t = torch.cat([pred_t, inverse_transform(pred_mask)])
        if mode == 'pseudo':
            # img_t = torch.cat([img_t, inverse_transform(image)])
            pass
        else:  # mode == 'val'
            img_t = torch.cat([img_t, inverse_transform(image)])
            gt_t = torch.cat([gt_t, inverse_transform(label)])
        confusion_metric.reset()

    if mode == 'pseudo':
        pseudo_name = np.array(name)
        pred_t = pred_t.detach().numpy()
        print(pseudo_name.shape, pred_t.shape,)
        np.savez_compressed(save_path,
                            image=pred_t,
                            image_name=pseudo_name)
    else:  # mode == 'val'
        print('Dice metrics', dice_metric.aggregate().item(), end='  ')
        print('F1/Dice Score:', torch.cat([i[0] for i in metrics]).mean().detach().cpu().numpy(), end='  ')
        print('Precision:', torch.cat([i[1] for i in metrics]).mean().detach().cpu().numpy(), end='  ')
        print('Recall:', torch.cat([i[2] for i in metrics]).mean().detach().cpu().numpy())
        pred_t = pred_t.detach().numpy()
        img_t = img_t.detach().numpy()
        gt_t = gt_t.detach().numpy()
        pseudo_name = np.array(name)
        np.savez_compressed(save_path,
                            pred=pred_t,
                            gt=gt_t,
                            raw=img_t,
                            name=pseudo_name)
        print(pred_t.shape, gt_t.shape, img_t.shape)
    print("\n\n=================== save {} at {}".format(mode, save_path))
    return

def select_reliable(args,model, dataset, save_path, model_path, name):
    models = []
    for checkpoint_n in sorted(os.listdir(model_path))[1:]:
        checkpoint_path = os.path.join(model_path, str(checkpoint_n))
        model = model.load_from_checkpoint(checkpoint_path, arch=args.model,
                                           in_channels=1, out_classes=1)
        model.eval().cuda(device=args.gpu)
        models.append(model)
    id_to_reliability = []
    with torch.no_grad():
        for id, (image, label) in enumerate(tqdm(dataset)):
            image = image.cuda(device=args.gpu)
            preds = []
            for model in models:
                pred = torch.sigmoid(model(image))
                pred_ = (pred > 0.5).float().detach()
                preds.append(pred_)
            f1_1 = metric_n(preds[0], preds[-1])
            f1_2 = metric_n(preds[1], preds[-1])
            mena_f1 = (f1_1 + f1_2)/2
            id_to_reliability.append([id, mena_f1])
    id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
    reliable_name, unreliable_name = [], []
    for i in id_to_reliability[:len(id_to_reliability) // 2]:
        id = i[0]
        for idx, j in enumerate(name):
            if id == idx:
                reliable_name.append([j, str(i[1])])
    for i in id_to_reliability[len(id_to_reliability) // 2:]:
        id = i[0]
        for idx, j in enumerate(name):
            if id == idx:
                unreliable_name.append([j, str(i[1])])
    with open(os.path.join(save_path, 'reliable_ids.txt'), 'w') as f:
        for elem in reliable_name:
            f.write(str(elem[0])+', '+str(elem[1])+'\n')
    with open(os.path.join(save_path, 'unreliable_ids.txt'), 'w') as f:
        for elem in unreliable_name:
            f.write(str(elem[0])+', '+str(elem[1])+'\n')
    return reliable_name, unreliable_name

def train(args, model, dataset):
    if mode_states == 'sup':
        batch_size = args.batch_size
    elif mode_states == 'semi':
        batch_size = args.semi_batch_size
    checkpoint_callback = ModelCheckpoint(monitor='valid_f1_score',
                                          save_top_k=1,
                                          save_weights_only=True,
                                          filename='best-{epoch:02d}-{valid_avg_loss:.5f}-{valid_f1_score:.5f}',
                                          verbose=False,
                                          mode='max')
    checkpoint_callback2 = ModelCheckpoint(monitor='valid_avg_loss',
                                           save_top_k=-1,
                                           save_weights_only=True,
                                           filename='step-{epoch:02d}-{valid_avg_loss:.5f}-{valid_f1_score:.5f}',
                                           verbose=False,
                                           mode='min',
                                           every_n_epochs=(args.epoch//3),)
    
    trainer = pl.Trainer(
        accelerator = "gpu",
        devices = [args.gpu],
        max_epochs = args.epoch,
        precision = 16,
        fast_dev_run = False,
        enable_progress_bar = True,
        callbacks = [checkpoint_callback, checkpoint_callback2],
        logger = CSVLogger(save_dir="./logs/" + args.dataset_name + "/", name=args.model + "_" + str(batch_size)),
    )
    time_start = datetime.now()
    trainer.fit(model, dataset)
    print("\nTraining execution time is: ", (datetime.now() - time_start))
    return

def main(args):
    data_path = os.path.join(args.dataset_path, args.dataset_name,'set3.npz')
    unlabeled_path = os.path.join(args.dataset_path, args.dataset_name_unlabeled,'set.npz')
    model = Model(arch=args.model, in_channels=1, out_channels=1, lr=args.learning_rate)
    print('\n================> Total stage 1/3: Supervised training on labeled images (SupOnly)')
    dataset = DataModule(data_dir = data_path, 
                         batch_size = args.batch_size,
                         numworkers = args.numworkers,
                         data_name = args.dataset_name,
                         is_patch = args.is_patch,
                         transform = get_st_augmentation)
    mode_states = 'sup'
    train(args, model, dataset)
    if args.st_train:
        print('\n\n\n================> Total stage 2/3: Pseudo labeling all unlabeled images')
        log_path, model_path, checkpoint_path, version, pseudo_path = path_all(args,args.batch_size, None, 'st_pseudo.npz')
        print(log_path,'\n' , str(version), '\n', model_path, '\n', checkpoint_path, '\n', pseudo_path)
        model = Model(arch=args.model, in_channels=1, out_channels=1, lr=args.learning_rate)
        model = model.load_from_checkpoint(checkpoint_path, arch=args.model, in_channels=1, out_channels=1, 
                                           lr=args.learning_rate)
        pseudo_data, pseudo_id = evaluate_data(args, unlabeled_path, batch_size=args.evaluate_batch_size)
        evaluate(args, model, pseudo_data, pseudo_path, mode='pseudo', name=pseudo_id)
        print('\n\n\n================> Total stage 3/3: Re-training on labeled and unlabeled images')
        model = Model(arch=args.model, in_channels=1, out_channels=1, lr=args.learning_rate)
        semi_data = DataModule_ST(data_dir = data_path, 
                                  data_dir_u= unlabeled_path,
                                  data_dir_p= pseudo_path,
                                  batch_size = args.semi_batch_size,
                                  numworkers = args.numworkers,
                                  data_name = args.dataset_name,
                                  is_patch = False,
                                  transform = get_stp_augmentation)
        mode_states = 'semi'
        train(args, model, semi_data)
        print('\n\n\n================> Model Evaluation on labeled image')
        log_path, model_path, checkpoint_path, version, pred_path = path_all(args,args.semi_batch_size, None, 'st_pred.npz')
        print(log_path,'\n' , str(version), '\n', model_path, '\n', checkpoint_path, '\n', pred_path)
        model = Model(arch=args.model, in_channels=1, out_channels=1, lr=args.learning_rate)
        model = model.load_from_checkpoint(checkpoint_path, arch=args.model, in_channels=1, out_channels=1, 
                                           lr=args.learning_rate)
        
        dataset.setup()
        val_data = dataset.val_dataloader()
        evaluate(args, model, val_data, pred_path, mode='val')
        return
    """
        ST++ framework with selective re-training
    """
    # <===================================== Select Reliable IDs =====================================>
    print('\n\n\n================> Total stage 2/6: Select reliable images for the 1st stage re-training')
    log_path, model_path, checkpoint_path, version, pseudo_path = path_all(args,args.batch_size, None, 'stplus_rel_pseudo.npz')
    print(log_path,'\n' , str(version), '\n', model_path, '\n', checkpoint_path, '\n', pseudo_path)
    pseudo_data, pseudo_id = evaluate_data(args, unlabeled_path, batch_size=1)
    model = Model(arch=args.model, in_channels=1, out_channels=1, lr=args.learning_rate)
    reliable, unreliable = select_reliable(args, model=model,dataset=pseudo_data, 
                                           save_path=model_path.replace('checkpoints', 'outdir'),
                                           model_path=model_path, name=pseudo_id)
    # <================================ Pseudo label reliable images =================================>
    print('\n\n\n================> Total stage 3/6: Pseudo labeling reliable images')
    model = Model(arch=args.model, in_channels=1, out_channels=1, lr=args.learning_rate)
    model = model.load_from_checkpoint(checkpoint_path, arch=args.model, in_channels=1, out_channels=1, 
                                       lr=args.learning_rate)
    pseudo_data, pseudo_id = evaluate_data(args, unlabeled_path, batch_size=args.evaluate_batch_size, rel=reliable)
    evaluate(args, model, pseudo_data, pseudo_path, mode='pseudo', name=pseudo_id)
    rel_pseudo_path = pseudo_path
    # <================================== The 1st stage re-training ==================================>
    print('\n\n\n================> Total stage 4/6: The 1st stage re-training on labeled and reliable unlabeled images')
    model = Model(arch=args.model, in_channels=1, out_channels=1, lr=args.learning_rate)
    semi_data = DataModule_ST(data_dir = data_path, 
                            data_dir_u= unlabeled_path,
                            data_dir_p= pseudo_path,
                            batch_size = args.semi_batch_size,
                            numworkers = args.numworkers,
                            data_name = args.dataset_name,
                            is_patch = False,
                            transform = get_stp_augmentation,
                            rel=True)
    mode_states = 'semi'
    train(args, model, semi_data)
    # <=============================== Pseudo label unreliable images ================================>
    print('\n\n\n================> Total stage 5/6: Pseudo labeling unreliable images')
    log_path, model_path, checkpoint_path, version, pseudo_path = path_all(args,args.semi_batch_size, None, 'stplus_unrel_pseudo.npz')
    print(log_path,'\n' , str(version), '\n', model_path, '\n', checkpoint_path, '\n', pseudo_path)
    model = Model(arch=args.model, in_channels=1, out_channels=1, lr=args.learning_rate)
    model = model.load_from_checkpoint(checkpoint_path, arch=args.model, in_channels=1, out_channels=1, 
                                       lr=args.learning_rate)
    pseudo_data, pseudo_id = evaluate_data(args, unlabeled_path, batch_size=args.evaluate_batch_size, rel=unreliable)
    evaluate(args, model, pseudo_data, pseudo_path, mode='pseudo', name=pseudo_id)
    unrel_pseudo_path = pseudo_path
    # <================================== The 2nd stage re-training ==================================>
    print('\n\n\n================> Total stage 6/6: The 2nd stage re-training on labeled and all unlabeled images')
    model = Model(arch=args.model, in_channels=1, out_channels=1, lr=args.learning_rate)
    pseudo_path = combine_pseudo_data(rel_pseudo_path, unrel_pseudo_path, 'stplus_all_pseudo.npz')
    semi_data = DataModule_ST(data_dir = data_path, 
                            data_dir_u= unlabeled_path,
                            data_dir_p= pseudo_path,
                            batch_size = args.semi_batch_size,
                            numworkers = args.numworkers,
                            data_name = args.dataset_name,
                            is_patch = False,
                            transform = get_stp_augmentation,
                            rel=True)
    train(args, model, semi_data)
    print('\n\n\n================> Model Evaluation on labeled image')
    log_path, model_path, checkpoint_path, version, pred_path = path_all(args,args.semi_batch_size, None, 'stplus_pred.npz')
    print(log_path,'\n' , str(version), '\n', model_path, '\n', checkpoint_path, '\n', pseudo_path)
    model = Model(arch=args.model, in_channels=1, out_channels=1, lr=args.learning_rate)
    model = model.load_from_checkpoint(checkpoint_path, arch=args.model, in_channels=1, out_channels=1, 
                                        lr=args.learning_rate)
    dataset.setup()
    val_data = dataset.val_dataloader()
    evaluate(args, model, val_data, pred_path, mode='val')
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model',default="UNet", type=str,
                        help='the name of model',
                        choices=['FR_UNet', 'R2AttU_Net', 'SegNet', 'NestedUNet', 'UNet_3Plus', 'AttU_Net', 'UNet'],
                        required=True)
    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('-sb', '--semi_batch_size', default=8, type=int)
    parser.add_argument('-eb', '--evaluate_batch_size', default=8, type=int)
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
    parser.add_argument('-dn', '--dataset_name', default="STARE", type=str,
                        help='the name of dataset', required=True)
    parser.add_argument('-dnu', '--dataset_name_unlabeled', default="STARE_u", type=str,
                        help='the name of unlabeled dataset')
    parser.add_argument('-st', '--st_train', action="store_true",
                        help='only self-training')
    args = parser.parse_args()
    
    main(args)
    
    ## command -- python train_st.py -m UNet -b 4 -dn STARE -e 1 -sb 4 -eb 16