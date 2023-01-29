from .model_arch import *
import torch
import pytorch_lightning as pl
from monai.metrics import DiceMetric, ConfusionMatrixMetric
import ssl
import segmentation_models_pytorch as smp_model
ssl._create_default_https_context = ssl._create_unverified_context

def get_model(arch, in_ch, out_ch,**kwargs):
    arch = str(arch).lower()
    if arch == "FR_UNet".lower():
        model = FR_UNet(num_classes=out_ch, num_channels=in_ch)
    elif arch == "CE_Net".lower():
        model = CE_Net(num_classes=out_ch, num_channels=in_ch)
    elif arch == "R2U_Net".lower():
        model = R2U_Net(img_ch=in_ch, output_ch=out_ch)
    elif arch == "R2AttU_Net".lower():
        model = R2AttU_Net(in_ch=in_ch, out_ch=out_ch)
    elif arch == "SegNet".lower():
        model = SegNet(input_nbr=in_ch, label_nbr=out_ch)
    elif arch == "NestedUNet".lower() or arch == "UNet++".lower():
        model = NestedUNet(in_channel=in_ch, out_channel=out_ch)
    elif arch == "UNet_3Plus".lower() or arch == "UNet+++".lower():
        model = UNet_3Plus(in_channels=in_ch, n_classes=out_ch)
    elif arch == "UNet".lower():
        model = UNet(n_channels=in_ch, n_classes=out_ch, bilinear=True)
    else:
        model = UNet(n_channels=in_ch, n_classes=out_ch, bilinear=True)
    return model


class Model(pl.LightningModule):
    def __init__(self, 
                 arch, 
                 encoder_name=None,  
                 encoder_weights="imagenet", 
                 in_channels=1, 
                 out_classes=1, 
                 lr=0.01,
                 **kwargs):
        super().__init__()
        self.aux_params = None
        # self.model = smp_model.create_model(
        #     arch, encoder_name=encoder_name, encoder_weights=encoder_weights,
        #     in_channels=in_channels, classes=out_classes,
        #     aux_params=self.aux_params
        # )
        
        # self.model = UNet(n_channels=in_channels,n_classes=out_classes,bilinear=True)
        self.model = get_model(arch, in_ch=in_channels, out_ch=out_classes)
        encoder_name=None
        try:
            # preprocessing parameters for image
            params = smp_model.encoders.get_preprocessing_params(encoder_name)
            self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
            self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
            self.pretrain = True
        except:
            self.pretrain = False
            # encoder_weights = "None"

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        # self.loss_fn = monai.losses.DiceLoss(sigmoid=True)
        # self.loss_fn = monai.losses.GeneralizedDiceLoss(sigmoid=True)
        # self.loss_fn = monai.losses.DiceCELoss(sigmoid=True)
        # self.loss_fn = monai.losses.GeneralizedDiceFocalLoss(sigmoid=True)

        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.confusion_metric = ConfusionMatrixMetric(include_background=False,
                                                      metric_name=['f1 score', 'precision', 'recall'],
                                                      reduction="mean", get_not_nans=False, compute_sample=True)

        self.lr = lr
        self.save_hyperparameters()

    def forward(self, image):
        # normalize image here
        if self.pretrain:
            image = (image - self.mean) / self.std
        else:
            image = image.float()

        if self.aux_params:
            mask, _ = self.model(image)
        else:
            mask = self.model(image)
            
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch

        assert image.ndim == 4
        h, w = image.shape[2:]
        # assert h % 32 == 0 and w % 32 == 0
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        if stage == "valid":
            prob_mask = logits_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()

            self.dice_metric(pred_mask, mask)
            self.confusion_metric(pred_mask, mask)

            dice_score = self.dice_metric.aggregate()
            f1_score, precision, recall = self.confusion_metric.aggregate()

            self.dice_metric.reset()
            self.confusion_metric.reset()
            logs = {'valid_loss': loss,
                    'valid_precision': precision,
                    'valid_recall': recall,
                    'valid_f1_score': f1_score,
                    'valid_dice_score': dice_score,
                    'lr': self.optimizer.param_groups[0]['lr']}
            self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            return {
                "loss": loss,
                "dice_score": dice_score,
                "f1_score": f1_score,
                "precision": precision,
                "recall": recall,
            }
        else:
            return {
                "loss": loss,
            }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metrics
        if stage == "valid":
            avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
            dice_score = torch.cat([x["dice_score"] for x in outputs]).mean()
            f1_score = torch.cat([x["f1_score"] for x in outputs]).mean()
            precision = torch.cat([x["precision"] for x in outputs]).mean()
            recall = torch.cat([x["recall"] for x in outputs]).mean()

            metrics = {
                f"{stage}_precision": precision,
                f"{stage}_recall": recall,
                f"{stage}_f1_score": f1_score,
                f"{stage}_dice_score": dice_score,
                f"{stage}_avg_loss": avg_loss,
            }
            self.log_dict(metrics, prog_bar=True, sync_dist=True)
        else:
            avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
            metrics = {
                f"{stage}_avg_loss": avg_loss,
            }
            self.log_dict(metrics, prog_bar=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5),
            'monitor': 'valid_avg_loss'}
        return {'optimizer': self.optimizer, 'lr_scheduler': self.lr_scheduler}
    
    
class Model_S3(pl.LightningModule):
    def __init__(self, 
                 arch, 
                 encoder_name=None,  
                 encoder_weights="imagenet", 
                 in_channels=1, 
                 out_classes=1, 
                 lr=0.01,
                 **kwargs):
        super().__init__()
        self.aux_params = None
        # self.model = smp_model.create_model(
        #     arch, encoder_name=encoder_name, encoder_weights=encoder_weights,
        #     in_channels=in_channels, classes=out_classes,
        #     aux_params=self.aux_params
        # )
        
        # self.model = UNet(n_channels=in_channels,n_classes=out_classes,bilinear=True)
        self.model = get_model(arch, in_ch=in_channels, out_ch=out_classes)
        encoder_name=None
        try:
            # preprocessing parameters for image
            params = smp_model.encoders.get_preprocessing_params(encoder_name)
            self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
            self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
            self.pretrain = True
        except:
            self.pretrain = False
            # encoder_weights = "None"

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.loss_fn_u = torch.nn.BCEWithLogitsLoss()
        # self.loss_fn = monai.losses.DiceLoss(sigmoid=True)
        # self.loss_fn = monai.losses.GeneralizedDiceLoss(sigmoid=True)
        # self.loss_fn = monai.losses.DiceCELoss(sigmoid=True)
        # self.loss_fn = monai.losses.GeneralizedDiceFocalLoss(sigmoid=True)

        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.confusion_metric = ConfusionMatrixMetric(include_background=False,
                                                      metric_name=['f1 score', 'precision', 'recall'],
                                                      reduction="mean", get_not_nans=False, compute_sample=True)

        self.lr = lr
        self.save_hyperparameters()

    def forward(self, image):
        # normalize image here
        if self.pretrain:
            image = (image - self.mean) / self.std
        else:
            image = image.float()

        if self.aux_params:
            mask, _ = self.model(image)
        else:
            mask = self.model(image)
            
        return mask

    def shared_step(self, batch, stage):
        if stage == "train": 
            image, mask = batch["labeled"]
            image_s, image_w = batch["unlabeled"]
        else:
            image, mask = batch


        logits_mask = self.forward(image)
        loss_l = self.loss_fn(logits_mask, mask)

        if stage == "train":
            pred_s = self.forward(image_s)
            pred_w = self.forward(image_w)
            mask_s = (torch.sigmoid(pred_w).ge(0.5)).to(dtype=torch.float32)
            loss_u = self.loss_fn_u(pred_s, torch.sigmoid(pred_w.detach()) * mask_s)
            loss = loss_l + loss_u
        else:
            loss = loss_l
        
        if stage == "valid":
            prob_mask = logits_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()

            self.dice_metric(pred_mask, mask)
            self.confusion_metric(pred_mask, mask)

            dice_score = self.dice_metric.aggregate()
            f1_score, precision, recall = self.confusion_metric.aggregate()

            self.dice_metric.reset()
            self.confusion_metric.reset()
            logs = {'valid_loss': loss,
                    'valid_precision': precision,
                    'valid_recall': recall,
                    'valid_f1_score': f1_score,
                    'valid_dice_score': dice_score,
                    'lr': self.optimizer.param_groups[0]['lr']}
            self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            return {
                "loss": loss,
                "dice_score": dice_score,
                "f1_score": f1_score,
                "precision": precision,
                "recall": recall,
            }
        else:
            return {
                "loss": loss,
            }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metrics
        if stage == "valid":
            avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
            dice_score = torch.cat([x["dice_score"] for x in outputs]).mean()
            f1_score = torch.cat([x["f1_score"] for x in outputs]).mean()
            precision = torch.cat([x["precision"] for x in outputs]).mean()
            recall = torch.cat([x["recall"] for x in outputs]).mean()

            metrics = {
                f"{stage}_precision": precision,
                f"{stage}_recall": recall,
                f"{stage}_f1_score": f1_score,
                f"{stage}_dice_score": dice_score,
                f"{stage}_avg_loss": avg_loss,
            }
            self.log_dict(metrics, prog_bar=True, sync_dist=True)
        else:
            avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
            metrics = {
                f"{stage}_avg_loss": avg_loss,
            }
            self.log_dict(metrics, prog_bar=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5),
            'monitor': 'valid_avg_loss'}
        return {'optimizer': self.optimizer, 'lr_scheduler': self.lr_scheduler}