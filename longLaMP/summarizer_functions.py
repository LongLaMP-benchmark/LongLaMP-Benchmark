from data.datasets import convert_to_gpt_dataset_for_summarization,convert_to_gpt_dataset,create_prompted_dataset_after_summarizing,create_prompted_dataset_with_styles_and_summaries,create_prompted_dataset_for_styles_and_summaries_per_profile,create_individual_extracted_profile_datset,create_prompted_dataset_stylistic_no_retrieval
from summarizer_gpt import run_summarizer
from prompts.utils import group_by_user_summary,load,combine_dataset_stylistic_and_summarized,combine_dataset_stylistic_and_summarized_per_profile_item
from gpt_stylistic_summarizer import run_stylistic

def content_only_summarizer(eval_dataset,file_path_to_gpt_output,task,flag = False,flag_only_summaries = False):
    if not flag:
        prompted_dataset = convert_to_gpt_dataset_for_summarization(eval_dataset)
        prompted_dataset = run_summarizer(prompted_dataset,file_path_to_gpt_output)
    else:
        try:
            print("loading the dataset summarized")
            prompted_dataset = load(file_path_to_gpt_output)    
        except FileNotFoundError:
                    print("The file does not exist.")
    prompted_dataset = group_by_user_summary(prompted_dataset)
    if flag_only_summaries: return prompted_dataset
    prompted_dataset = create_prompted_dataset_after_summarizing(prompted_dataset,task)
    return prompted_dataset

def stylistic_only_extraction(eval_dataset,file_path_to_save_gpt_output,flag=False):
    if not flag:
        prompted_dataset = convert_to_gpt_dataset(eval_dataset)
        prompted_dataset = run_stylistic(prompted_dataset,file_path_to_save_gpt_output)
    else:
        try:
            print("loading the dataset stylized")
            prompted_dataset = load(file_path_to_save_gpt_output)
        except FileNotFoundError:
                    print("The file does not exist.")
    return prompted_dataset

def style_extraction_with_inp(eval_dataset,file_path_to_save_gpt_output,task,flag=False):
     prompted_dataset = stylistic_only_extraction(eval_dataset,file_path_to_save_gpt_output,flag)
     prompted_dataset = create_prompted_dataset_stylistic_no_retrieval(prompted_dataset,task)
     return prompted_dataset

def stylistic_summarization(eval_dataset_summarization,eval_dataset_stylistic,file_path_to_save_gpt_output_summarization,file_path_to_save_gpt_output_stylistic,task,flag_summarization=False,flag_stylization=False):
    prompted_dataset_summarized = content_only_summarizer(eval_dataset_summarization,file_path_to_save_gpt_output_summarization,task,flag_summarization,flag_only_summaries = True)
    prompted_dataset_stylistic = stylistic_only_extraction(eval_dataset_stylistic,file_path_to_save_gpt_output_stylistic,flag_stylization)

    combined_dataset = combine_dataset_stylistic_and_summarized(prompted_dataset_summarized,prompted_dataset_stylistic)

    prompted_dataset = create_prompted_dataset_with_styles_and_summaries(combined_dataset,task)

    return prompted_dataset

def stylistic_summarization_per_profile(eval_dataset_per_profile,eval_dataset_stylistic,file_path_to_save_gpt_per_profile,file_path_to_save_gpt_output_stylistic,task,flag_profile=False,flag_stylized= False):
    if not flag_profile:
        selected_profiles = create_individual_extracted_profile_datset(eval_dataset_per_profile)
    else:
        try:
            selected_profiles = load(file_path_to_save_gpt_per_profile)
        except FileNotFoundError:
             print("The file does not exist.")
    stylized_outputs = stylistic_only_extraction(eval_dataset_stylistic,file_path_to_save_gpt_output_stylistic,flag_stylized)
    combined_datset = combine_dataset_stylistic_and_summarized_per_profile_item(selected_profiles,stylized_outputs)
    prompted_dataset_for_summarization = create_prompted_dataset_for_styles_and_summaries_per_profile(combined_datset,task)
    summary_results = run_summarizer(prompted_dataset_for_summarization,file_path_to_save_gpt_per_profile)
    grouped_users_by_summary = group_by_user_summary(summary_results)
    prompted_dataset = create_prompted_dataset_after_summarizing(grouped_users_by_summary,task)
    return prompted_dataset




      

    


