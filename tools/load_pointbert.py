import torch
import torch.nn as nn
from pointllm.utils import *


class pointbert_cls(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.emb_dim = 768  # default

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.num_classes = num_classes
        self.classifier = nn.Linear(self.emb_dim, self.num_classes)

    def forward(self, batch_X: torch.Tensor):
        embedding = self.backbone(batch_X)
        logit = self.classifier(embedding)

        return logit


def load_pointbert(config_path: str, ckpt_path: str, use_color: bool = False):
    from pointllm.model import PointTransformer
    point_bert_config = cfg_from_yaml_file(config_path)

    if use_color:
        point_bert_config.model.point_dims = 6
    use_max_pool = getattr(point_bert_config.model,
                           "use_max_pool", False)  # * default is false

    pointbert = PointTransformer(
        point_bert_config.model, use_max_pool=use_max_pool)
    print(f"Using {pointbert.point_dims} dim of points.")

    point_backbone_config = {
        "point_cloud_dim": point_bert_config.model.point_dims,
        "backbone_output_dim": point_bert_config.model.trans_dim if not use_max_pool else point_bert_config.model.trans_dim * 2,
        # "project_output_dim": self.config.hidden_size,
        # * number of output features, with cls token
        "point_token_len": point_bert_config.model.num_group + 1 if not use_max_pool else 1,
        # "mm_use_point_start_end": self.config.mm_use_point_start_end,
        "projection_hidden_layer": point_bert_config.model.get('projection_hidden_layer', 0),
        "use_max_pool": use_max_pool
    }
    if point_bert_config.model.get('projection_hidden_layer', 0) > 0:
        # a list
        point_backbone_config["projection_hidden_dim"] = point_bert_config.model.projection_hidden_dim

    print(
        f"Use max pool is {use_max_pool}. Number of point token is {point_backbone_config['point_token_len']}.")

    pointbert_prefix = "model.point_backbone."
    ckpt = torch.load(ckpt_path)
    ckpt = {
        k[len(pointbert_prefix):]: v for k, v in ckpt.items() if k.startswith(pointbert_prefix)
    }
    pointbert.load_state_dict(ckpt)

    return pointbert


if __name__ == "__main__":
    load_pointbert("pointllm/model/pointbert/PointTransformer_8192point_2layer.yaml",
                   "RunsenXu/PointLLM_13B_v1.2/pytorch_model-00006-of-00006.bin", use_color=True)
