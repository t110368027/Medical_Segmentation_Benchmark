import os
import numpy as np
from monai.metrics import ConfusionMatrixMetric

def inverse_transform(source_tensor):
    """
    source_tensor : B, C, H, W

    target_tensor : B, H, W, C

    :param source_tensor: torch.Tensor()
    :return: target_tensor: torch.Tensor()
    """
    assert source_tensor.ndim == 4
    source_tensor = source_tensor.detach().cpu()
    target_tensor = source_tensor.permute(0, 2, 3, 1)
    return target_tensor

def metric_n(pred,gt):
    confusion_metric = ConfusionMatrixMetric(include_background=False,
                                             metric_name=['f1 score','precision','recall'],
                                             reduction="mean", get_not_nans=False,compute_sample=False)
    confusion_metric(pred.detach(), gt.detach())
    f1_score = confusion_metric.aggregate()[0].detach().cpu().numpy()[0]
    confusion_metric.reset()
    return f1_score

def path_all(args, batch_size, version_n, save_name):
    log_path = ("./logs/{}/{}_{}".format(args.dataset_name,
                                            args.model,
                                            str(batch_size)))
    if version_n:
        version = str(version_n)
    else:
        version = str(sorted(os.listdir(log_path))[-1])
    model_path = os.path.join(log_path + '/' + version + '/checkpoints')
    best_model = str(sorted(os.listdir(model_path))[0])
    checkpoint_path = os.path.join(model_path, best_model)
    if not os.path.exists(model_path.replace('checkpoints', 'outdir')):
        os.mkdir(model_path.replace('checkpoints', 'outdir'))
    if save_name:
        pseudo_path = os.path.join(model_path.replace('checkpoints', 'outdir'), save_name)
        return log_path, model_path, checkpoint_path, version, pseudo_path
    else:
        return log_path, model_path, checkpoint_path, version
    
def combine_pseudo_data(reliable, unreliable, save_path):
    save_path = unreliable.replace(unreliable.split('\\')[1], save_path)
    print('reliable: {}, unreliable: {},\n save_path: {}'.format(reliable, unreliable, save_path))
    with np.load(reliable, allow_pickle=True) as f:
        rel_train_u, rel_train_u_id = f['image'], f['image_name']
    with np.load(unreliable, allow_pickle=True) as f:
        unrel_train_u, unrel_train_u_id = f['image'], f['image_name']
    pseudo_all_img = np.concatenate((rel_train_u, unrel_train_u))
    pseudo_all_id = np.concatenate((rel_train_u_id, unrel_train_u_id))
    np.savez_compressed(save_path,
                        image=pseudo_all_img,
                        image_name=pseudo_all_id)
    return save_path