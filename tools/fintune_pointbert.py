import argparse
from load_pointbert import load_pointbert, pointbert_cls
from load_modelnet40 import ModelNetDataset


def main(config_path: str, ckpt_path: str, data_path: str):
    print(f"Config file: {config_path}")
    print(f"Weights file: {ckpt_path}")
    print(f"Dataset path: {data_path}")

    pointbert = load_pointbert(config_path,
                               ckpt_path, use_color=True)

    pointbert = pointbert_cls(pointbert, 10)

    TEST_DATASET = ModelNetDataset(
        root=data_path,
        npoint=8192,
        split='test',
        normal_channel=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training or evaluation.")
    parser.add_argument("--config-path", type=str, default="pointllm/model/pointbert/PointTransformer_8192point_2layer.yaml",
                        help="Path to config file")
    parser.add_argument("--ckpt-path", type=str, default="RunsenXu/PointLLM_13B_v1.2/pytorch_model-00006-of-00006.bin",
                        help="Path to model weights")
    parser.add_argument("--data-path", type=str, default="data/modelnet40_normal_resampled",
                        help="Path to dataset")

    args = parser.parse_args()
    main(args.config_path, args.ckpt_path, args.data_path)
