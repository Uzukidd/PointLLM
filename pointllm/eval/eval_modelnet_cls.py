import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from pointllm.conversation import SeparatorStyle, conv_templates
from pointllm.data import CustonModelNet, ModelNet
from pointllm.eval.evaluator import start_evaluation
from pointllm.eval.localLLaMA_evaluator import localLLaMA_close_set_cls_evaluator
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.utils import disable_torch_init

PROMPT_LISTS = ["What is this?", "This is an object of "]


def init_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    # * print the model_name (get the basename)
    print(f"[INFO] Model name: {os.path.basename(model_name)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=False,
        use_cache=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    conv_mode = "vicuna_v1_1"

    conv = conv_templates[conv_mode].copy()

    return model, tokenizer, conv


def load_dataset(
    dataset: str,
    data_path: str,
    use_adv: bool,
    config_path,
    split,
    subset_nums,
    use_color,
):
    if dataset == "ModelNet40":
        print(f"Loading {split} split of ModelNet datasets.")
        dataset = ModelNet(
            config_path=config_path,
            split=split,
            subset_nums=subset_nums,
            use_color=use_color,
        )
        print("Done!")
    elif dataset == "CustomDataset":
        print(f"Loading {split} split of Custom datasets.")
        dataset = CustonModelNet(data_path, use_adv=use_adv, use_color=True)
        print("Done!")
    return dataset


def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    assert (
        shuffle is False
    ), "Since we using the index of ModelNet as Object ID when evaluation \
        so shuffle shoudl be False and should always set random seed."
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataloader


def generate_outputs(
    model,
    tokenizer,
    input_ids,
    point_clouds,
    stopping_criteria,
    do_sample=True,
    temperature=1.0,
    top_k=50,
    max_length=2048,
    top_p=0.95,
):
    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            point_clouds=point_clouds,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            max_length=max_length,
            top_p=top_p,
            stopping_criteria=[stopping_criteria],
        )  # * B, L'

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )
    outputs = [output.strip() for output in outputs]

    return outputs


def start_generation(
    model, tokenizer, conv, dataloader, prompt_index, output_dir, output_file
):
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    qs = PROMPT_LISTS[prompt_index]

    results = {"prompt": qs}

    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config["point_token_len"]
    default_point_patch_token = point_backbone_config["default_point_patch_token"]
    default_point_start_token = point_backbone_config["default_point_start_token"]
    default_point_end_token = point_backbone_config["default_point_end_token"]
    mm_use_point_start_end = point_backbone_config["mm_use_point_start_end"]

    if mm_use_point_start_end:
        qs = (
            default_point_start_token
            + default_point_patch_token * point_token_len
            + default_point_end_token
            + "\n"
            + qs
        )
    else:
        qs = default_point_patch_token * point_token_len + "\n" + qs

    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])

    input_ids_ = torch.as_tensor(inputs.input_ids).cuda()  # * tensor of 1, L

    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_)

    responses = []
    for epoch in tqdm(range(5)):
        for batch in tqdm(dataloader):
            point_clouds = (
                batch["point_clouds"].cuda().to(model.dtype)
            )  # * tensor of B, N, C(3)
            labels = batch["labels"]
            label_names = batch["label_names"]
            indice = batch["indice"]

            batchsize = point_clouds.shape[0]

            input_ids = input_ids_.repeat(batchsize, 1)  # * tensor of B, L

            outputs = generate_outputs(
                model, tokenizer, input_ids, point_clouds, stopping_criteria
            )  # List of str, length is B

            # saving results
            for index, output, label, label_name in zip(
                indice, outputs, labels, label_names
            ):
                responses.append(
                    {
                        "object_id": index.item(),
                        "ground_truth": label.item(),
                        "model_output": output,
                        "label_name": label_name,
                    }
                )

    results["results"] = responses

    os.makedirs(output_dir, exist_ok=True)
    # save the results to a JSON file
    with open(os.path.join(output_dir, output_file), "w") as fp:
        json.dump(results, fp, indent=2)

    # * print info
    print(f"Saved results to {os.path.join(output_dir, output_file)}")

    return results


def main(args):
    # * ouptut
    args.output_dir = os.path.join(args.model_name, "evaluation")

    # * output file
    args.output_file = f"{args.dataset}_classification_prompt{args.prompt_index}.json"
    args.output_file_path = os.path.join(args.output_dir, args.output_file)

    # * First inferencing, then evaluate
    if not os.path.exists(args.output_file_path):
        # * need to generate results first
        dataset = load_dataset(
            args.dataset,
            args.data_path,
            args.use_adv,
            config_path=None,
            split=args.split,
            subset_nums=args.subset_nums,
            use_color=args.use_color,
        )  # * defalut config
        dataloader = get_dataloader(
            dataset, args.batch_size, args.shuffle, args.num_workers
        )

        model, tokenizer, conv = init_model(args)

        # * ouptut
        print(f"[INFO] Start generating results for {args.output_file}.")
        results = start_generation(
            model,
            tokenizer,
            conv,
            dataloader,
            args.prompt_index,
            args.output_dir,
            args.output_file,
        )

        # * release model and tokenizer, and release cuda memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
    else:
        # * directly load the results
        print(f"[INFO] {args.output_file_path} already exists, directly loading...")
        with open(args.output_file_path, "r") as fp:
            results = json.load(fp)

    # * evaluation file
    evaluated_output_file = args.output_file.replace(
        ".json", f"_evaluated_{args.gpt_type}.json"
    )
    # * start evaluation
    if args.start_eval:
        start_evaluation(
            results,
            output_dir=args.output_dir,
            output_file=evaluated_output_file,
            eval_type="localllama-modelnet-close-set-classification",
            model_type=args.gpt_type,
            parallel=False,
            num_workers=20,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="RunsenXu/PointLLM_13B_v1.2")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ModelNet40",
        choices=["ModelNet40", "CustomDataset"],
    )

    # * dataset type
    parser.add_argument(
        "--epoch", type=int, default=5, help="Whether to use adversarial data"
    )
    parser.add_argument(
        "--use_adv", action="store_true", help="Whether to use adversarial data"
    )
    parser.add_argument(
        "--data_path", type=str, default="test", help="path to the dataset"
    )
    parser.add_argument("--split", type=str, default="test", help="train or test.")
    parser.add_argument("--use_color", action="store_true", default=True)

    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument(
        "--subset_nums", type=int, default=-1
    )  # * only use "subset_nums" of samples, mainly for debug

    # * evaluation setting
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--start_eval", action="store_true", default=False)
    parser.add_argument(
        "--gpt_type",
        type=str,
        default="Meta-Llama-3.1-8B-Instruct",
        choices=[
            "Meta-Llama-3.1-8B-Instruct",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-1106",
            "gpt-4-0613",
            "gpt-4-1106-preview",
        ],
        help="Type of the model used to evaluate.",
    )

    args = parser.parse_args()

    main(args)
