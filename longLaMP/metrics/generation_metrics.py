import numpy as np
import evaluate

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def create_metric_bleu_rouge_meteor(tokenizer):
    bleu_metric = evaluate.load("sacrebleu")
    rouge_metric = evaluate.load('rouge')
    meteor_metric = evaluate.load('meteor')
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id) #TODO: check
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result_bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_meteor = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu" : result_bleu["score"], "rouge-1" : result_rouge["rouge1"], "rouge-2" : result_rouge["rouge2"], "rouge-L" : result_rouge["rougeL"], "rouge-LSum" : result_rouge["rougeLsum"], "meteor" : result_meteor['meteor']}
        return result
    return compute_metrics

def compute_metrics(decoded_preds, decoded_labels):
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    bleu_metric = evaluate.load("sacrebleu")
    rouge_metric = evaluate.load('rouge')
    meteor_metric = evaluate.load('meteor')
    result_bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result_rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result_meteor = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu" : result_bleu["score"], "rouge-1" : result_rouge["rouge1"], "rouge-2" : result_rouge["rouge2"], "rouge-L" : result_rouge["rougeL"], "rouge-LSum" : result_rouge["rougeLsum"], "meteor" : result_meteor['meteor']}
    return result

def evaluate_data(data,name):
    decoded_preds,decoded_labels = [],[]
    for d in data:
        if d["generated_text"] and d["output"]:
            decoded_preds.append(d["generated_text"])
            decoded_labels.append(d["output"])
    rouge_metric = compute_metrics(decoded_preds, decoded_labels)
    with open("metrics.txt", "a") as f: #TODO: parameterize
        f.write(name + ":" + str(rouge_metric) +'\n')
    return rouge_metric
