from transformers import AutoTokenizer
from metrics.generation_metrics import create_metric_bleu_rouge_meteor,create_metric_bleu_rouge_meteor_chatgpt
from data.datasets_llama import get_all_labels, GeneralSeq2SeqDataset, create_preprocessor, convert_to_hf_dataset,convert_to_llama_dataset,create_preprocessor_chatgpt
from prompts.prompts_llama import create_prompt_generator
from test_llama import perform, run, run_vllm
import argparse
from evaluation import evaluate_data
import json
import os
import logging
import time

parser = argparse.ArgumentParser()
parser.add_argument("--name", required = True)
parser.add_argument("--validation_data", required = True)
parser.add_argument("--tokenizer", required = False)
parser.add_argument("--task", required = True)
parser.add_argument("--output_dir", required = True)
parser.add_argument("--output_dir_retrieved", required = True)
parser.add_argument("--use_profile", action = "store_false")
parser.add_argument("--retriever", default = "bm25")
parser.add_argument("--num_support_profile", type = int, default = 1)
parser.add_argument("--is_ranked", action = "store_true")
parser.add_argument("--cache_dir", default = "./cache")
parser.add_argument("--max_length", type = int, default = 512)


if __name__ == "__main__":
    #TODO: change this!
    logging.basicConfig(filename='/home/sviswanathan_umass_edu/llama_vllm_topic_user_test_np.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    opts = parser.parse_args()
    retriever,task = opts.retriever,opts.task
    logging.info("Creating prompt")
    start = time.process_time()
    if opts.use_profile:
        prompt_generator, contriver = create_prompt_generator(opts.num_support_profile, opts.retriever, opts.is_ranked, opts.max_length, tokenizer = None)
        logging.info("Got prompt")
    else:
        prompt_generator, contriver = None, None
    logging.info("use_profile: "+str(opts.use_profile))
    if task == "generation_abstract":
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
    elif task == "topic_writing":
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
    elif task == "product_review_writing":
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
    elif task == "email_generation":
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
    logging.info("Created eval dataset")

    logging.info("Getting into convert")
    prompted_dataset = convert_to_llama_dataset(eval_dataset)
    logging.info("Converted to llama dataset")
    logging.info("Added to retrieved json file")
    results = run_vllm(prompted_dataset,opts.output_dir)
    logging.info("Got results")
    metrics = evaluate_data(results,opts.name)
    logging.info("Got metrics")
    logging.info(f"{round((time.process_time() - start), 2)}s ")
    print(f"{round((time.process_time() - start), 2)}s ")



