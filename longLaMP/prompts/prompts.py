from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
from prompts.utils import extract_strings_between_quotes, extract_after_article, extract_after_review, extract_after_paper, add_string_after_title, extract_input_string,extract_before_bullets,extract_after_colon
from prompts.contriver_retriever import retrieve_top_k_with_contriver
import random
import logging
import traceback

def generate_query_for_topic_writing(inp, profile):
    corpus = [f'{x["input"]} {x["output"]}' for x in profile]
    query = extract_input_string(inp)
    return corpus, query

def generate_query_for_product_review_writing(inp, profile):
    corpus = [f'{x["overall"]} {x["summary"]} {x["description"]} {x["reviewText"]}' for x in profile]
    query = inp
    return corpus, query

def generation_abstract_query_corpus_maker(inp, profile):
    corpus = [f'{x["title"]} {x["abstract"]}' for x in profile]
    query = extract_before_bullets(inp)
    return corpus, query

def generate_query_for_email_generation(inp, profile):
    corpus = [f'{x["text"]}' for x in profile]
    query = extract_after_colon(inp)
    return corpus, query

def create_generation_topic_prompt(inp, profile, max_length, tokenizer):
    prompts = []
    for p in profile:
        tokens = tokenizer(p["output"], max_length=max_length, truncation=True)
        content = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
        prompt = f'"{p["input"]}" is a summary for "{content}" '
        prompts.append(prompt)
    return f'{", and ".join(prompts)}. Following the given patterns, {inp}'

def create_generation_product_review_prompt(inp, profile, max_length, tokenizer):
    prompts = []
    for p in profile:
        tokens = tokenizer(p["reviewText"], max_length=max_length, truncation=True)
        review_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
        prompt = f'"{p["overall"]}" is a rating for the product with description "{p["description"]}". "{p["summary"]}" is summary for "{review_text}" '
        prompts.append(prompt)
    return f'{", and ".join(prompts)}. Following the given patterns {inp}'

def create_abstract_paper_prompt(inp, profile, max_length, tokenizer):
    prompts = []
    for p in profile:
        extracted_text = extract_first_750_words(p["abstract"])
        tokens = tokenizer(extracted_text, max_length=max_length, truncation=True)
        abstract_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0] #TODO:verify
        promptStr =  f'"{abstract_text}" is the abstract for the title "{p["title"]}"'
        prompts.append(promptStr)
    return f'{", and ".join(prompts)}. Use the above abstracts as context to understand the style and language of the user and, {inp}'

def extract_first_750_words(text):
    words = text.split()
    first_750_words = words[:750]
    extracted_text = ' '.join(first_750_words)
    return extracted_text

def create_email_generation_prompt(inp, profile, max_length, tokenizer):
    prompts = []
    for p in profile:
        tokens = tokenizer(p["text"], max_length=max_length, truncation=True)
        text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
        prompt = f'"{p["title"]}" is the title for "{text}" '
        prompts.append(prompt)
    return f'{", and ".join(prompts)}. {inp}'

def create_prompt_generator(num_retrieve, time_tally, ret_type = "bm25", is_ranked = False, summarizer_technique=None, max_length = 512, tokenizer = None, output_file_paths= None):
    contriver = None
    if ret_type == "contriver" and not is_ranked:
        tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        contriver = AutoModel.from_pretrained('facebook/contriever').to("cuda:0")
        contriver.eval()

    def prompt(inp, profile, task, output=None):
        if task == "topic_writing":
            corpus, query = generate_query_for_topic_writing(inp, profile)
        elif task == "product_review_writing":
            corpus, query = generate_query_for_product_review_writing(inp, profile)
        elif task == "generation_abstract":
            corpus, query = generation_abstract_query_corpus_maker(inp, profile)
        elif task == "email_generation":
            corpus, query = generate_query_for_email_generation(inp, profile)
        
        if not is_ranked:
            if ret_type == "bm25":
                tokenized_corpus = [x.split() for x in corpus]
                bm25 = BM25Okapi(tokenized_corpus)
                tokenized_query = query.split()
                selected_profs = bm25.get_top_n(tokenized_query, profile, n=num_retrieve)
            elif ret_type == "contriver":
                selected_profs = retrieve_top_k_with_contriver(contriver, tokenizer, corpus, profile, query, num_retrieve)
            elif ret_type == "random":
                selected_profs = random.choices(profile, k = num_retrieve)
            elif ret_type == "rec":
                selected_profs = profile[-num_retrieve:][::-1]
        else:
            selected_profs = profile[:num_retrieve]
        try:
            max_len_prompt = None
            if task == "topic_writing":
                return create_generation_topic_prompt(inp, selected_profs,max_len_prompt, tokenizer)
            elif task == "product_review_writing":
                return create_generation_product_review_prompt(inp, selected_profs,max_len_prompt, tokenizer)
            elif task == "generation_abstract":
                return create_abstract_paper_prompt(inp, selected_profs,max_len_prompt, tokenizer)
            elif task == "email_generation":
                return create_email_generation_prompt(inp, selected_profs,max_len_prompt, tokenizer)
        except Exception as e:
            tb = e.__traceback__
            lineno = traceback.extract_tb(tb)[-1].lineno
            print(f'Exception is {e}, at line number: {lineno}')
    return prompt, contriver