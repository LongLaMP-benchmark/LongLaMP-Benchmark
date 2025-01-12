from transformers import AutoTokenizer
from data.datasets import  GeneralSeq2SeqDataset,convert_to_gpt_dataset
from prompts.prompts import create_prompt_generator
from run_gpt import run
import argparse
from metrics.generation_metrics import evaluate_data
import json
import os
import time
from summarizer_functions import *



parser = argparse.ArgumentParser()
parser.add_argument("--name", required = True)  # name if the task. Used for storing the metrics
parser.add_argument("--validation_data", required = True) # path to the validation dataset
parser.add_argument("--tokenizer", required = False) # see if tokenization is required
parser.add_argument("--task", required = True) # name of the task we are performing
parser.add_argument("--output_dir", required = True) # final output directory for the generated output
parser.add_argument("--metrics_file", required = True) # file for metrics
parser.add_argument("--output_dir_retrieved", required = False) # file path to save the retrieved files.
parser.add_argument("--output_dir_retrieved_profiles", required = False) #file path to save the retrieved profiles
parser.add_argument("--output_dir_summarized",required=False) # file path to save the summarized files
parser.add_argument("--output_dir_stylized",required = False) # file path to save the stylized files
parser.add_argument("--use_profile", action = "store_false") # to check if we need to retrieve anything for not
parser.add_argument("--use_stylistic_summarizer", action = "store_true") # flag for using summarization
parser.add_argument("--retriever", default = "bm25") # type of retriever
parser.add_argument("--num_support_profile", type = int, default = 1) # to check the number of profiles that we need 
parser.add_argument("--is_ranked", action = "store_true") # to see if we need to rank or just randomly retrieve
parser.add_argument("--is_stylized", action = "store_true") # flag to check if something is already stylized
parser.add_argument("--is_summarized", action = "store_true") # flag to check if something is already summarized
parser.add_argument("--is_retrieved", action = "store_true")
parser.add_argument("--summary_task",required = False) # name of the summarization task
parser.add_argument("--needs_retreival",action = "store_false") # if we need to retrieve anything
parser.add_argument("--is_ranked_file", action = "store_true") # file with ranked entries
parser.add_argument("--cache_dir", default = "./cache") # path to chache directory
parser.add_argument("--max_length", type = int, default = 512) # the max length 

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    
    tokenizer = None
    time_tally = {"total_retrieval_time":0,"average_retrieval_time":0,"running_prompting_time":[],"total_prompting_time":0, "average_prompting_time" : 0}
    start_time_all = time.time()
    opts = parser.parse_args()
    style_directory,summary_directory,summary_task,prompted_dataset,retrieved_profiles = None,None,None,None,opts.output_dir_retrieved_profiles
    if not opts.needs_retreival: 
        retrieved_profiles = opts.output_dir_retrieved_profiles
    #if opts.output_dir_summarized : summary_directory = opts.output_dir_summarized
    if opts.summary_task : summary_task = opts.summary_task
    retriever,task = opts.retriever,opts.task
    if opts.use_profile:
        prompt_generator, contriver = create_prompt_generator(opts.num_support_profile,opts.retriever,opts.use_stylistic_summarizer,opts.is_ranked, needs_retreival = opts.needs_retreival,retrieval_path = retrieved_profiles,tokenizer = None,time_tally = time_tally)
    else:
        prompt_generator, contriver = None, None
    eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile,task,retrieved_profiles,opts.use_stylistic_summarizer,opts.needs_retreival,summary_task,prompt_generator)

    if contriver:
        contriver = contriver.to("cpu")


    if opts.use_stylistic_summarizer:
        if summary_task =="content_only_summarization":
            prompted_dataset = content_only_summarizer(eval_dataset,opts.output_dir_summarized,task,flag=opts.is_summarized)
        elif summary_task == "stylistic_only_extraction":
            prompted_dataset = style_extraction_with_inp(eval_dataset,opts.output_dir_stylized,task,flag=opts.is_stylized)
        elif summary_task  == "stylistic_summarization":
            eval_dataset_summarization = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile,task,retrieved_profiles,opts.use_stylistic_summarizer,opts.needs_retreival,"content_only_summarization",prompt_generator)
            prompted_dataset = stylistic_summarization(eval_dataset_summarization,eval_dataset,opts.output_dir_summarized,opts.output_dir_stylized,task,flag_summarization=opts.is_summarized,flag_stylization=opts.is_stylized)
        elif summary_task == "stylistic_summarization_element_wise":
            eval_dataset_stylized = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile,task,opts.output_dir_summarized,"stylistic_only_extraction",prompt_generator)
            prompted_dataset = stylistic_summarization_per_profile(eval_dataset,eval_dataset_stylized,opts.output_dir_retrieved,opts.output_dir_stylized,task,flag_profile=opts.is_retrieved,flag_stylized= opts.is_stylized)
    else:
        prompted_dataset = convert_to_gpt_dataset(eval_dataset)
    
    if prompted_dataset:
        if not opts.is_ranked_file:
            end_time_just_retrieval = time.time()
            with open(opts.output_dir_retrieved, "w") as f:
                json.dump(prompted_dataset,f,indent=4)
        else : 
            with open(opts.output_dir_retrieved,"r") as f:
                prompted_dataset = json.load(f)
        results = run(prompted_dataset,opts.output_dir,time_tally)
        metrics = evaluate_data(results,opts.metrics_file,opts.name)
        end_time_all = time.time()
        time_tally["total_prompting_time"] += end_time_all - start_time_all
        time_tally["average_prompting_time"] = time_tally["total_prompting_time"] / len(time_tally["running_prompting_time"])
        print(time_tally)



