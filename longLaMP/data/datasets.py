from torch.utils.data import Dataset
import json
import datasets
import logging

def create_preprocessor(tokenizer, max_length):
    def preprocess_dataset(examples):
        inputs = []
        for example in examples["source"]:
            escaped_example = example.replace('"', '\\"')
            inputs.append(escaped_example)

        targets = [example.replace('"', '\\"') for example in examples["target"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
        return model_inputs
    return preprocess_dataset

def create_preprocessor_chatgpt(tokenizer, max_length):
    def preprocess_dataset(examples):
        inputs = [example for example in examples["source"]]
        targets = [example for example in examples["target"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
        model_inputs = tokenizer.batch_decode(model_inputs['input_ids'], skip_special_tokens=True)
        return {"chatgpt_inputs" : model_inputs}
    return preprocess_dataset

def convert_to_hf_dataset(dataset):
    def gen():
        for idx in range(len(dataset)):
            yield dataset[idx]
    return datasets.Dataset.from_generator(gen)

def convert_to_llama_dataset(dataset):
    logging.info("Inside convert function")
    new_prompted_dataset = []
    logging.info("dataset length: "+str(dataset.__len__()))
    logging.info("dataset first item: "+str(dataset.__getitem__(0)))
    for idx in range(dataset.__len__()):
        new_prompted_dataset.append(dataset.__getitem__(idx))
    logging.info("Finished converting")
    return new_prompted_dataset

def convert_to_gpt_dataset(dataset):
    new_prompted_dataset = []
    for idx in range(dataset.__len__()):
        new_prompted_dataset.append(dataset.__getitem__(idx))
    return new_prompted_dataset

class GeneralSeq2SeqDataset(Dataset):

    def __init__(self, data_addr, use_profile, task, create_prompt = None) -> None:
        super().__init__()
            
        items = []
        with open(data_addr) as file:
            self.data = json.load(file)
        self.use_profile = use_profile
        self.task = task
        assert not (use_profile ^ (create_prompt != None)), "You should provide a prompt maker function when you use profile"
        self.create_prompt = create_prompt

    def __getitem__(self, index):
        if self.use_profile:
            return {
                "source" : self.create_prompt(self.data[index]['input'], self.data[index]['profile'], self.task, self.data[index]['output']),
                "target" : self.data[index]['output']
            }
        else:
            return {
                "source" : self.data[index]['input'],
                "target" : self.data[index]['output']
            }
    
    def __len__(self):
        return len(self.data)
