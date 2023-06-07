"""DelicacyNet model"""

import torch
import torch.nn as nn
from utils.misc import *
from .EFEBlock import build_backbone
from .EncoderToDecoder import build_EncoderToDecoder
from .loss import *
import pdfplumber
import pandas as pd
class DelicacyNet(nn.Module):
    def __init__(self, backbone, EncoderToDecoder, num_classes, dim):
        super(DelicacyNet, self).__init__()
        self.EncoderToDecoder = EncoderToDecoder
        hidden_dim = EncoderToDecoder.d_model
        self.query_embed = nn.Embedding(1, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        if isinstance(x, (list, torch.Tensor)):
            x = nested_tensor_from_tensor_list(x)
        
        features, pos = self.backbone(x)

        src, mask = features.decompose()
        assert mask is not None
        hs = self.EncoderToDecoder(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.mlp_head(hs)
        #outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1]} #, 'pred_boxes': outputs_coord[-1]}
        return out


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 2000
    device = torch.device(args.device)

    backbone = build_backbone(args)

    EncoderToDecoder = build_EncoderToDecoder(args)

    model = DelicacyNet(
        backbone,
        EncoderToDecoder,
        num_classes=num_classes,
        dim=512
    )

    '''
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    '''
    weights = get_weights(args.device)
    criterion = WmaeLoss(weights, args.device)
    criterion.to(device)

    return model, criterion

def get_weights(device):
    read_path = './data/Supplementary tables.pdf'
    pdf = pdfplumber.open(read_path)

    result_df = pd.DataFrame()
    for page in pdf.pages:
        table = page.extract_table()
        df_detail = pd.DataFrame(table[1:], columns=table[0])
        # 合并每页的数据集
        result_df = pd.concat([df_detail, result_df], ignore_index=True)
    result_df.columns = ['Index','Category','Number','index','category','number']
    result = result_df[['Index',"Category",'Number']]
    result_df=result_df[['index','category','number']]
    result_df.columns=['Index','Category','Number']
    result = pd.concat([result, result_df], ignore_index=True).sort_values("Index").reset_index(drop=True)
    result['Number'] = pd.to_numeric(result['Number'].str.replace(',', ''))
    num_samples = sum(result['Number']) 
    weights = []
    for i in range(len(result)):
        weights.append(num_samples/2000/result.loc[i,'Number'])
    return torch.tensor(weights).to(device)