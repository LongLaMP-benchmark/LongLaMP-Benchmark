from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
from prompts.utils import extract_strings_between_quotes, extract_after_article, extract_after_review, extract_after_paper, add_string_after_title, extract_input_string,extract_before_bullets
from prompts.contriver_retriever import retrieve_top_k_with_contriver
import random
import logging

def generate_query_for_email_generation(inp, profile):
    corpus = [f'{x["text"]}' for x in profile]
    query = extract_input_string(inp)
    return corpus, query

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

def create_generation_topic_prompt_llama(inp,profile):
    prompts = []
    for p in profile:
        prompt = f'"{p["input"]}" is a summary for "{p["output"]}" '
        prompts.append(prompt)
    return f'{", and ".join(prompts)}. Following the given patterns {inp}'

def extract_first_100_words(text):
    words = text.split()    
    first_100_words = words[:100]    
    extracted_text = ' '.join(first_100_words)
    
    return extracted_text

def create_generation_product_prompt_llama(inp,profile):
    prompts = []
    prompts.append("Below are a list of review texts by the reviewer:")
    for p in profile:
        extracted_text = extract_first_100_words(p["reviewText"])
        prompt = f'"{extracted_text}"'
        prompts.append(prompt)
    return f'{", and ".join(prompts)}. Use the above list of review texts by the reviewer as context to understand the writing style and language of the reviewer and, {inp}'

def create_generation_email_prompt_llama(inp,profile):
    prompts = []
    for p in profile:
        prompt = f'"{p["title"]}" is the title for "{p["text"]}" '
        prompts.append(prompt)
    return f'{", and ".join(prompts)}. Following the given patterns {inp}'

def create_generation_product_review_prompt(inp, profile, max_length, tokenizer):
    prompts = []
    for p in profile:
        tokens = tokenizer(p["reviewText"], max_length=max_length, truncation=True)
        new_review_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
        prompt = f'"{p["overall"]}" is a rating for the product with description "{p["description"]}". "{p["summary"]}" is summary for "{new_review_text}" '
        prompts.append(prompt)
    return f'{", and ".join(prompts)}. Following the given patterns {inp}'

def create_abstract_paper_prompt(inp, profile, max_length, tokenizer):
    prompts = []
    for p in profile:
        extracted_text = extract_first_750_words(p["abstract"])
        promptStr =  f'"{extracted_text}" is the abstract for the title "{p["title"]}"'
        prompts.append(promptStr)
    return f'{", and ".join(prompts)}. Use the above abstracts as context to understand the style and language of the user and, {inp}'

def extract_first_750_words(text):
    words = text.split()
    first_750_words = words[:750]
    extracted_text = ' '.join(first_750_words)
    return extracted_text

def create_prompt_generator(num_retrieve, ret_type = "bm25", is_ranked = False, max_length = 512, tokenizer = None):
    contriver = None
    if ret_type == "contriver" and not is_ranked:
        tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        contriver = AutoModel.from_pretrained('facebook/contriever').to("cuda:0")
        contriver.eval()

    def prompt(inp, profile, task):
        if task == "topic_writing":
            corpus, query = generate_query_for_topic_writing(inp, profile)
        elif task == "product_review_writing":
            corpus, query = generate_query_for_product_review_writing(inp, profile)
        elif task == "generation_abstract":
            corpus, query = generation_abstract_query_corpus_maker(inp, profile)
        elif task == "email_generation":
            corpus, query = generate_query_for_email_generation(inp, profile)
        
        logging.info("ret type:"+str(ret_type))
        if ret_type == "bm25":
            tokenized_corpus = [x.split() for x in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query.split()
            selected_profs = bm25.get_top_n(tokenized_query, profile, n=num_retrieve)
            print("BM25")
            logging.info("BM25")
        elif ret_type == "contriver":
            selected_profs = retrieve_top_k_with_contriver(contriver, tokenizer, corpus, profile, query, num_retrieve)
            print("Contriver")
            logging.info("Contriver")
        elif ret_type == "random":
            selected_profs = random.choices(profile, k = num_retrieve)
        elif ret_type == "rec":
            selected_profs = profile[-num_retrieve:][::-1]
        else:
            selected_profs = profile[:num_retrieve]

        factor = 0.6
        while True:
            try:
                if task == "topic_writing":
                    return create_generation_topic_prompt_llama(inp, selected_profs)
                elif task == "product_review_writing":
                    return create_generation_product_prompt_llama(inp, selected_profs)
                elif task == "generation_abstract":
                    return create_abstract_paper_prompt(inp, selected_profs)
                elif task == "email_generation":
                    return create_generation_email_prompt_llama(inp, selected_profs)
            except Exception as e:
                factor -= 0.1
                if factor < 0:
                    # print('input length exceeded')
                    pass
    return prompt, contriver