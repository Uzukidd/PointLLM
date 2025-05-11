import json
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from tqdm import tqdm
import re
import random
random.seed(0)

llama_close_set_cls_prompt = """Given the following free-form description of a 3D object, please determine the most probable class index from the following 40 available categories. Make your best-educated guess based on the information provided. If the description already contains a valid index, then the index should be selected. If it contains more than one valid index, then randomly select one index which starts from 0 (specify your reason). If there is no valid index and it cannot be inferred from the information, return '-1#NA#Cannot infer'.
Categories:
{candidate_lists}
Reply with the format of 'index#class#short reason (no more than 10 words)'.

Examples:
Input: This is a 3D object model of a cartoon white truck.
Output: 7#car#Closest match to 'car' in categories which index is 7.

Input: A green leaf in a flower pot.
Output: 26#plant#The primary subject 'leaf' directly indicates a plant which index is 26.

Input: It's difficult to determine the exact type of this object due to insufficient details. But it seems to be like a piece of furniture.
Output: 33#table#Randomly select one kind of furniture (table which index is 33) from the list.

Input:  I cannot determine the specific type of the object without additional information or context.
Output: -1#NA#Cannot infer.

Now analyze the following:
Input: """


class localLLaMA_close_set_cls_evaluator():
    def __init__(self, inputs, output_dir, output_file, model_type="Meta-Llama-3.1-8B-Instruct"):
        """
        Args:
            inputs: A dictionary containing the results of the evaluation. It contains two keys: "results" and "prompt".
                "prompt": str
                "results": [
                    {
                        "object_id": str,
                        "model_output": str,
                        "ground_truth": str
                    }
                ]
        """
        print("-" * 80)
        print("Initializing LocalLLaMA Evaluator...")
        # * contains two keys: "results" and "prompt"
        self.results = inputs['results']
        self.inference_prompt = inputs['prompt']  # * used to prompt PointLLM
        self.correct_predictions = 0
        self.total_predictions = 0
        self.invalid_responses = 0
        self.response_data = []  # to save all the response data by openaigpt
        self.model_type = model_type

        self.prompt_tokens = 0
        self.completion_tokens = 0

        # self.default_chat_parameters = {
        #     "model": model_type,
        #     "temperature": 1,
        #     "top_p": 1,
        #     "max_tokens": 2048
        # }

        # print(self.default_chat_parameters)
        self.output_dir = output_dir
        self.output_file = output_file
        self.temp_output_file = self.output_file.replace(
            ".json", "_processed_temp.json")

        self.prompt = llama_close_set_cls_prompt
        self.invalid_correct_predictions = 0  # * random choice and correct coincidently
        # * import category names
        try:
            # * load a txt files of category names
            # * i.e. pointllm/data/modelnet_config/modelnet40_shape_names_modified.txt
            catfile = os.path.join(os.path.dirname(
                __file__), '../data/modelnet_config/modelnet40_shape_names_modified.txt')
            self.candidate_lists_names = [line.strip() for line in open(
                catfile)]  # * list of category names
        except:
            print(
                f"Current categories file is {catfile}. Need to move the category file to pointllm/eval/configs/.")

        # * make the prompt
        candidate_lists = [f'{cat}' for i,
                           cat in enumerate(self.candidate_lists_names)]
        self.num_categories = len(candidate_lists)
        self.candidate_lists = '\n'.join(candidate_lists)
        self.prompt = self.prompt.format(
            num_categories=self.num_categories, candidate_lists=self.candidate_lists) + "{model_output}\nOutput: "
        print(self.prompt)
        # initialize local LLaMA
        self.init_localLLaMA()

    def init_localLLaMA(self):
        model_path = os.path.join("./llama/", self.model_type)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, device_map="auto")

    def resume_processing(self):
        processed_results_path = os.path.join(
            self.output_dir, self.temp_output_file)
        if os.path.exists(processed_results_path):
            print("-" * 80)
            # * print resuming
            print(f"Resuming processing...")
            print(
                f"Loading processed results from {processed_results_path}...")
            with open(processed_results_path, "r") as f:
                saved_results = json.load(f)
            self.correct_predictions = saved_results["correct_predictions"]
            self.total_predictions = saved_results["total_predictions"]
            self.invalid_responses = saved_results["invalid_responses"]
            self.response_data = saved_results["results"]
            self.prompt_tokens = saved_results["prompt_tokens"]
            self.completion_tokens = saved_results["completion_tokens"]

            print(f"Processed results: {len(self.response_data)}")
            # * print the length of all the data
            print(f"Total results: {len(self.results)}")

            # * remove processed data
            processed_ids = [d['object_id'] for d in self.response_data]
            self.results = [
                r for r in self.results if r['object_id'] not in processed_ids]

            print(f"Remaining results: {len(self.results)}")

    def remove_temp_file(self):
        processed_results_path = os.path.join(
            self.output_dir, self.temp_output_file)
        if os.path.exists(processed_results_path):
            os.remove(processed_results_path)
            print("-" * 80)
            print(f"Removed Temporary file {processed_results_path}")

    def parse_response_evaluate(self, gpt_response, ground_truth, label_name):
        """
        Argument:
            gpt_response: str, index#label#short_reason
            groud_truth: int
        """
        # * use regular expression to extract
        pattern = r'(\d+#[^#]*#.*$)'
        match = re.search(pattern, gpt_response)

        gpt_response = match.group(1) if match else gpt_response

        gpt_response = gpt_response.strip()
        gpt_response_list = gpt_response.split('#')

        cls_result = gpt_response_list[0]
        cls_label = gpt_response_list[1] if len(gpt_response_list) > 1 else ""
        reason = gpt_response_list[2] if len(gpt_response_list) > 2 else ""

        try:
            # * convert to int
            cls_result = int(cls_result)
            if cls_result not in range(self.num_categories) or cls_label == "NA":
                # * not valid range
                cls_result = -1
        except ValueError:
            # print(cls_result)
            print(f"Error: unale to parse {gpt_response}.")
            # import pdb;pdb.set_trace()
            cls_result = -1

        if cls_result == -1:
            # * random choose one index from 0 to self.num_categories
            cls_result = random.choice(range(self.num_categories))
            cls_label = "INVALID"
            reason = gpt_response

            self.invalid_responses += 1

        accuracy = 1 if cls_result == ground_truth or cls_label == label_name else 0

        return accuracy, cls_result, cls_label, reason

    def evaluate_result(self, result):
        object_id = result['object_id']
        ground_truth = result['ground_truth']
        model_output = result['model_output']
        label_name = result['label_name']
        messages = [
            {"role": "user", "content": self.prompt.format(model_output=model_output)}]

        # gpt_response = self.openaigpt.safe_chat_complete(
        #     messages, content_only=False)

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt").cuda()
        input_len = inputs.shape[1]
        generated_ids = self.model.generate(
            inputs,
            temperature= 1,
            top_p= 1,
            max_new_tokens=2048,
        )
        new_tokens = generated_ids[0, input_len:]
        generated_text = self.tokenizer.decode(
            new_tokens, skip_special_tokens=True)

        response = generated_text
        accuracy, cls_result, cls_label, reason = self.parse_response_evaluate(
            # return 0, "INVALID", gpt_response if not valid
            response, ground_truth, label_name)

        return object_id, model_output, ground_truth, accuracy, cls_result, reason

    def evaluate(self):

        self.resume_processing()

        print('-' * 80)
        print("Starting single-thread evaluation...")
        results = self.results

        try:
            for result in tqdm(results):
                object_id, model_output, ground_truth, accuracy, cls_result, reason = self.evaluate_result(
                    result)
                self.correct_predictions += accuracy
                self.total_predictions += 1
                # self.prompt_tokens += prompt_tokens
                # self.completion_tokens += completion_tokens

                # save the object_id, model_output, ground_truth, gpt_cls_result and gpt_reason for each result
                self.response_data.append({
                    'object_id': object_id,
                    'ground_truth': ground_truth,
                    'model_output': model_output,
                    'gpt_cls_result': cls_result,
                    'gpt_reason': reason
                })
                # print(accuracy)
                # print(self.response_data[-1])

            print("Evaluation finished.")

            self.save_results()
            self.print_results()
            self.remove_temp_file()
        except KeyboardInterrupt as e:
            print(
                f"Error {e} occurred during parallel evaluation. Saving processed results to temporary file...")
            self.save_results(is_temp=True)
            exit()

    def parallel_evaluate(self, num_workers=20):

        self.resume_processing()

        print('-' * 80)
        print("Starting parallel evaluation...")
        results = self.results

        try:
            with Pool(num_workers) as pool:
                with tqdm(total=len(results)) as pbar:  # create a progress bar
                    for object_id, model_output, ground_truth, accuracy, cls_result, reason, prompt_tokens, completion_tokens in pool.imap_unordered(self.evaluate_result, results):
                        self.correct_predictions += accuracy
                        self.total_predictions += 1
                        self.prompt_tokens += prompt_tokens
                        self.completion_tokens += completion_tokens

                        if cls_result == 'INVALID':
                            self.invalid_responses += 1

                        # save the object_id, model_output, ground_truth, gpt_cls_result and gpt_reason for each result
                        self.response_data.append({
                            'object_id': object_id,
                            'ground_truth': ground_truth,
                            'model_output': model_output,
                            'gpt_cls_result': cls_result,
                            'gpt_reason': reason
                        })

                        pbar.update()  # update the progress bar

            print("Parallel evaluation finished.")

            self.save_results()
            self.print_results()
            self.remove_temp_file()

        except KeyboardInterrupt as e:
            print(
                f"Error {e} occurred during parallel evaluation. Saving processed results to temporary file...")
            self.save_results(is_temp=True)
            exit()

    def save_results(self, is_temp=False):
        if is_temp:
            output_path = os.path.join(self.output_dir, self.temp_output_file)
        else:
            output_path = os.path.join(self.output_dir, self.output_file)
        if self.total_predictions - self.invalid_responses == 0:
            accuracy = 0  # * no results and get error
        else:
            accuracy = self.correct_predictions / \
                (self.total_predictions - self.invalid_responses) * 100
        with open(output_path, 'w') as f:
            results_to_save = {
                'inference_prompt': self.inference_prompt,
                'prompt': self.prompt,
                'accuracy': f"{accuracy:.2f}%",
                'total_predictions': self.total_predictions,
                'correct_predictions': self.correct_predictions,
                'invalid_responses': self.invalid_responses,
                'prompt_tokens': self.prompt_tokens,
                'completion_tokens': self.completion_tokens,
                'GPT_cost': self.get_costs(),
                'results': self.response_data,
            }
            json.dump(results_to_save, f, indent=2)

        print(f"Results saved to {output_path}")
        # * print the length of saved results
        print(f"Saved {len(self.response_data)} results in total.")

    def print_results(self):
        print('-' * 80)
        if self.total_predictions - self.invalid_responses == 0:
            accuracy = 0  # * no results and get error
        else:
            accuracy = self.correct_predictions / \
                (self.total_predictions - self.invalid_responses) * 100
        print("Results:")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Total Predictions: {self.total_predictions}")
        print(f"Correct Predictions: {self.correct_predictions}")
        print(f"Invalid Responses: {self.invalid_responses}")
        # self.print_costs()

    def print_costs(self):
        print(
            f"Prompt Tokens Price: {self.prompt_tokens * self.price_1k_prompt_tokens / 1000:.2f} USD")
        print(
            f"Completion Tokens Price: {self.completion_tokens * self.price_1k_completion_tokens / 1000:.2f} USD")

    def get_costs(self):
        return 0
