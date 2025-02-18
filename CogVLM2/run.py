import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

MODEL_PATH = "THUDM/cogvlm2-llama3-chat-19B"
TORCH_TYPE = torch.bfloat16
device = 'cuda:0'

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=TORCH_TYPE,
    trust_remote_code=True,
    device_map=device,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    low_cpu_mem_usage=True
).eval()

def recur_move_to(item, tgt, criterion_func):
    if criterion_func(item):
        device_copy = item.to(tgt)
        return device_copy
    elif isinstance(item, list):
        return [recur_move_to(v, tgt, criterion_func) for v in item]
    elif isinstance(item, tuple):
        return tuple([recur_move_to(v, tgt, criterion_func) for v in item])
    elif isinstance(item, dict):
        return {k: recur_move_to(v, tgt, criterion_func) for k, v in item.items()}
    else:
        return item

def collate_fn(features, tokenizer) -> dict:
    images = [feature.pop('images', None) for feature in features if 'images' in feature]
    tokenizer.pad_token = tokenizer.eos_token
    max_length = max(len(feature['input_ids']) for feature in features)
    def pad_to_max_length(feature, max_length):
        padding_length = max_length - len(feature['input_ids'])
        feature['input_ids'] = torch.cat([feature['input_ids'], torch.full((padding_length,), tokenizer.pad_token_id)])
        feature['token_type_ids'] = torch.cat([feature['token_type_ids'], torch.zeros(padding_length, dtype=torch.long)])
        feature['attention_mask'] = torch.cat([feature['attention_mask'], torch.zeros(padding_length, dtype=torch.long)])
        if feature['labels'] is not None:
            feature['labels'] = torch.cat([feature['labels'], torch.full((padding_length,), tokenizer.pad_token_id)])
        else:
            feature['labels'] = torch.full((max_length,), tokenizer.pad_token_id)
        return feature
    features = [pad_to_max_length(feature, max_length) for feature in features]
    batch = {
        key: torch.stack([feature[key] for feature in features])
        for key in features[0].keys()
    }
    if images:
        batch['images'] = images
    return batch


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="CogVLM2 Image Captioning")
    parser.add_argument("--image_folder", type=str, default="./flat_dataset/run_1/sample", help="Path to the folder containing images.")
    parser.add_argument("--output_json", type=str, default="./captions.json", help="Path to save the output JSON file.")
    parser.add_argument("--batch_size", type=int, default=3, help="Batch size for processing images.")
    parser.add_argument("--query", type=str, default="Describe this image in detail, and the description should be between 15 to 80 words.")
    args = parser.parse_args()

    # Load images from the specified folder
    image_folder = args.image_folder

    data = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                data.append({"image": os.path.join(root, file)})

    length = len(data)
    batch_size = args.batch_size
    query = args.query

    # Dictionary to store image names as keys and captions as values
    captions_dict = {}

    for idx in range(0, length, batch_size):
        i_list = []
        for i in range(batch_size):
            if idx + i < length:
                i_list.append(data[idx + i])
            else:
                break

        input_sample_list = []
        start = time.time()
        for i in i_list:
            image = Image.open(i["image"]).convert('RGB')
            input_sample = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image], template_version='chat')
            input_sample_list.append(input_sample)
        print(f"Prepare input time: {time.time() - start}")

        start = time.time()
        input_batch = collate_fn(input_sample_list, tokenizer)
        input_batch = recur_move_to(input_batch, device, lambda x: isinstance(x, torch.Tensor))
        input_batch = recur_move_to(input_batch, torch.bfloat16, lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x))
        print(f"Prepare batch time: {time.time() - start}")

        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
            "top_k": 1,
        }

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**input_batch, **gen_kwargs)
            outputs = outputs[:, input_batch['input_ids'].shape[1]:]
            outputs = tokenizer.batch_decode(outputs)

        outlist = [output.split("<|end_of_text|>")[0].strip() for output in outputs]
        print(outlist)
        print(f"Generate time: {time.time() - start}")

        # Add captions to the dictionary
        for i, caption in zip(i_list, outlist):
            image_name = os.path.basename(i["image"])  # Extract the image name
            captions_dict[image_name] = caption

    # Save the captions dictionary to a JSON file
    json_file_path = args.output_json
    with open(json_file_path, "w") as json_file:
        json.dump(captions_dict, json_file, indent=4)

    print(f"Captions saved to {json_file_path}")