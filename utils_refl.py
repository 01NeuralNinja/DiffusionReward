import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.archs.vgg_arch import VGGFeatureExtractor
from pytorch_wavelets import DTCWTForward
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image

    BICUBIC = Image.BICUBIC
import torchvision
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer


def setup_reward_model(cfg, weight_dtype, accelerator):
    """
    设置并配置reward model，包括模型创建、权重加载和参数设置

    Args:
        cfg: 配置对象
        weight_dtype: 权重数据类型
        accelerator: Accelerator对象

    Returns:
        tuple: (reward_model, tokenizer, normalize)
    """
    # 创建reward model
    model_name = "ViT-H-14"
    reward_model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_name,
        'laion2B-s32B-b79K',
        precision=weight_dtype,
        device=accelerator.device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )

    # 获取tokenizer
    tokenizer = get_tokenizer(model_name)

    # 加载预训练权重
    checkpoint_path = cfg.train.facereward_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print('reward_load----------------')
    print(reward_model.load_state_dict(checkpoint['state_dict'], strict=True))
    reward_model = reward_model.to(accelerator.device)

    # 配置模型参数
    reward_model.set_grad_checkpointing(True)
    reward_model.train()
    reward_model.lock_image_tower(
        unlocked_groups=20,
        freeze_bn_stats=False)
    reward_model.lock_text_tower(
        unlocked_layers=11,
        freeze_layer_norm=False)

    # 设置图像归一化
    normalize = torchvision.transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    target_size = 224

    return reward_model, tokenizer, normalize, target_size
# 添加对抗训练的损失函数
def hps_loss_ru(text_logits, labels):
    device = text_logits.device
    text_0_logits, text_1_logits = text_logits.chunk(2, dim=-1)
    label_0, label_1 = labels.chunk(2, dim=-1)
    index = torch.arange(text_0_logits.shape[0], device=device, dtype=torch.long)
    text_0_logits = text_0_logits[index, index]
    text_1_logits = text_1_logits[index, index]
    text_logits = torch.stack([text_0_logits, text_1_logits], dim=-1)
    text_0_labels = torch.zeros(text_logits.shape[0], device=device, dtype=torch.long)
    text_1_labels = text_0_labels + 1
    text_0_loss = torch.nn.functional.cross_entropy(text_logits, text_0_labels, reduction="none")
    text_1_loss = torch.nn.functional.cross_entropy(text_logits, text_1_labels, reduction="none")
    text_loss = label_0 * text_0_loss + label_1 * text_1_loss
    text_loss = text_loss.sum()
    return text_loss


_reduction_modes = ['none', 'mean', 'sum']
import functools
def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are 'none', 'mean' and 'sum'.

    Returns:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    else:
        return loss.sum()
def weight_reduce_loss(loss, weight=None, reduction='mean'):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights. Default: None.
        reduction (str): Same as built-in losses of PyTorch. Options are
            'none', 'mean' and 'sum'. Default: 'mean'.

    Returns:
        Tensor: Loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if weight is not specified or reduction is sum, just reduce the loss
    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    # if reduction is mean, then compute mean over weight region
    elif reduction == 'mean':
        if weight.size(1) > 1:
            weight = weight.sum()
        else:
            weight = weight.sum() * loss.size(1)
        loss = loss.sum() / weight

    return loss

def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.5000)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, reduction='sum')
    tensor(3.)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction)
        return loss

    return wrapper
@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target) ** 2 + eps)



class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)









class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss



class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1',
                 gram_norm=True):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)
        self.gram_norm = gram_norm

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(gt_features[k])) * \
                                  self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h).to(torch.float32)
        if self.gram_norm:
            normalized_features = F.layer_norm(features, features.shape[1:])
            normalized_features_t = normalized_features.transpose(1, 2)
            gram = normalized_features.bmm(normalized_features_t) / (c * h * w)
        else:
            features_t = features.transpose(1, 2)
            gram = features.bmm(features_t) / (c * h * w)
        return gram.to(torch.float16)



class DWTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, lowq=True, highq=False, reduction='mean'):
        super(DWTLoss, self).__init__()

        self.loss = L1Loss(reduction=reduction)
        self.loss_weight = loss_weight
        self.lowq = lowq
        self.highq = highq
        self.dwt = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')

    def forward(self, x, y):
        x_dwt_l, x_dwt_h = self.dwt(x)
        y_dwt_l, y_dwt_h = self.dwt(y)
        # import pdb; pdb.set_trace()
        loss = 0
        if self.lowq:
            loss += self.loss(x_dwt_l, y_dwt_l)
        if self.highq:
            for x_dwt_h_sub, y_dwt_h_sub in zip(x_dwt_h, y_dwt_h):
                loss += self.loss(x_dwt_h_sub, y_dwt_h_sub)
        return loss * self.loss_weight



class KLLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='batchmean'):
        super(KLLoss, self).__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss = nn.KLDivLoss(reduction=reduction)

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * self.loss(pred, target)





