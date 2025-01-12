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
        max_tokens = 2048,
        temperature=0.6
    )
    return response.choices[0].message.content

def get_stylistic_elements(data):
    encoding = tiktoken.get_encoding("cl100k_base")
    prompt_stylistic = data["source"]
    input = data["input"]
    output = data["target"]
    if len(encoding.encode(prompt_stylistic)) >= 2000:
        print("input context over limt")
        return {
                   "stylistic elements" : None,
                    "input": input,
                    "target":output
                }
    max_retries = 100
    trials = 0
    counter = 0
    while trials < max_retries:
        trials +=1
        try:
            result_stylistic = get_result(prompt_stylistic)
            counter = 0
            break
        except Exception as e:
            counter += 1
            if "This model's maximum context" in e.__str__():
                return {
                   "stylistic elements" : None,
                   "input": input,
                    "target":output
                }
            if "The response was filtered due to the prompt" in e.__str__():
                return {
                   "stylistic elements" : None,
                   "input": input,
                    "target":output
                }
            print("error", e)
        if counter == 10:
                time.sleep(60)

    if trials == max_retries:
        print(f"Maximum retries ({max_retries}) reached. Skipping this instance.")
        return {
                    "stylistic elements" : None,
                    "input": input,
                    "target":output
                }
    print("sucess")

    obj = {
        "stylistic elements" : result_stylistic,
        "input": input,
        "target":output
    }
    return obj

def run_stylistic(dataset,filepath):
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(get_stylistic_elements, d) for d in dataset]
    completed, _ = concurrent.futures.wait(futures)
    results = [future.result() for future in completed]
    with open(filepath, "w") as outfile: 
        json.dump(results,outfile,indent=4)
    return results
    
   
           
  