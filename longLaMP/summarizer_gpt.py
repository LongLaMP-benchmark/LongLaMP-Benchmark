import json
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor
import concurrent
import time
import os
import argparse
import tiktoken

parser = argparse.ArgumentParser()

#Configure your open ai endpoint here.
client = AzureOpenAI(
  azure_endpoint = "",
  api_key='',
  api_version=""
)

def get_result(inp):
    response = client.chat.completions.create(
        model="gpt-35-turbo-viet",
        messages = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": inp}],
        max_tokens = 850,
        temperature=0.2
    )
    return response.choices[0].message.content


def perform(data):
    encoding = tiktoken.get_encoding("cl100k_base")

    start_time_prompt = time.time()
    out = data["target"]
    prompt = data["source"]
    input = data["input"]
    if len(encoding.encode(prompt)) >= 3000:
        print("input context over limt")
        return {
                   "summary" : None,
                   "target" : out,
                   "input":input
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
                   "summary" : None,
                   "target" : out,
                   "input":input
                }
            if "The response was filtered due to the prompt" in e.__str__():
                return {
                   "summary" : None,
                   "target" : out,
                   "input":input
                }
            print("error", e)
        if counter == 10:
                time.sleep(60)

    if trials == max_retries:
        print(f"Maximum retries ({max_retries}) reached. Skipping this instance.")
        return {
                   "summary" : None,
                   "target" : out,
                   "input":input
                }
    print("sucess")
    end_time_prompt = time.time()
    obj = {
        "summary" : result,
        "target" : out,
        "input":input
    }
    return obj

def run_summarizer(dataset,filepath):
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(perform, d) for d in dataset]
    completed, _ = concurrent.futures.wait(futures)
    results = [future.result() for future in completed]
    with open(filepath, "w") as outfile: 
        json.dump(results,outfile,indent=4)
    return results