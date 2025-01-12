import json
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor
import concurrent
import time
import os
import argparse
import tiktoken

parser = argparse.ArgumentParser()

#Configure your GPT endpoint
client = AzureOpenAI(
  azure_endpoint = "",
  api_key='',
  api_version=""
)


def get_result(inp):
    response = client.chat.completions.create(
        model="gpt-35-turbo-viet",
        messages = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": inp}],
        max_tokens = 2048,
        temperature=0.6
    )
    return response.choices[0].message.content

def perform_abstract_fix(data,time_tally):
    start_time_prompt = time.time()
    out = data["target"]
    prompt = data["source"]
    counter = 0
    while True:
        try:
            result = get_result(prompt)
            counter = 0
            break
        except Exception as e:
            counter += 1
            if "This model's maximum context" in e.__str__():
                return {
                    "generated_abstract" : None,
                    "output" : out
                }
            print("error", e)
            if counter == 10:
                time.sleep(60)

    print("sucess")
    end_time_prompt = time.time()
    time_tally["running_prompting_time"].append(end_time_prompt - start_time_prompt)
    obj = {
        "generated_abstract" : result,
        "output" : out
    }
    return obj

def run_abstract_fix(dataset,filepath,time_tally):
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(perform_abstract_fix, d,time_tally) for d in dataset]
    completed, _ = concurrent.futures.wait(futures)
    results = [future.result() for future in completed]
    with open(filepath, "w") as outfile: 
        json.dump(results,outfile,indent=4)
    return results

def perform(data,time_tally):
    encoding = tiktoken.get_encoding("cl100k_base")

    start_time_prompt = time.time()
    out = data["target"]
    prompt = data["source"]
    if len(encoding.encode(prompt)) >= 5000:
        print("input context over limt")
        return {
                   "generated_abstract" : None,
                   "output" : out
                }
    counter = 0
    max_retries = 100
    trials = 0
    while trials < max_retries:
        trials +=1
        try:
            result = get_result(prompt)
            counter = 0
            break
        except Exception as e:
            counter += 1
            if "This model's maximum context" in e.__str__():
                return {
                   "generated_abstract" : None,
                   "output" : out
                }
            if "The response was filtered due to the prompt" in e.__str__():
                return {
                   "generated_abstract" : None,
                   "output" : out
                }
            print("error", e)
        if counter == 10:
                time.sleep(60)

    if trials == max_retries:
        print(f"Maximum retries ({max_retries}) reached. Skipping this instance.")
        return {
                   "generated_abstract" : None,
                   "output" : out
                }
    print("sucess")
    end_time_prompt = time.time()
    time_tally["running_prompting_time"].append(end_time_prompt - start_time_prompt)
    obj = {
        "generated_abstract" : result,
        "output" : out
    }
    return obj

def run(dataset,filepath,time_tally):
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(perform, d,time_tally) for d in dataset]
    completed, _ = concurrent.futures.wait(futures)
    results = [future.result() for future in completed]
    with open(filepath, "w") as outfile: 
        json.dump(results,outfile,indent=4)
    return results