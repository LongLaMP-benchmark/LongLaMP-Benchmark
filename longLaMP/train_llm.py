from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.data.data_collator import DataCollatorForSeq2Seq
import argparse
from metrics.generation_metrics import create_metric_bleu_rouge_meteor
from data.datasets import get_all_labels, GeneralSeq2SeqDataset, create_preprocessor, convert_to_hf_dataset
from prompts.prompts import create_prompt_generator

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", required=False)
parser.add_argument("--train_data", required = True)
parser.add_argument("--validation_data", required = True)
parser.add_argument("--test_data", required = True)
parser.add_argument("--model_name", required = True)
parser.add_argument("--task", required = True)
parser.add_argument("--output_dir", required = True)
parser.add_argument("--retriever", default = "bm25")
parser.add_argument("--use_profile", action = "store_true")
parser.add_argument("--is_ranked", action = "store_true")
parser.add_argument("--max_length", type = int, default = 256)
parser.add_argument("--generation_max_length", type = int, default = 128)
parser.add_argument("--per_device_batch_size", type = int, default = 16)
parser.add_argument("--learning_rate", type = float, default = 5e-5)
parser.add_argument("--weight_decay", type = float, default = 0.0001)
parser.add_argument("--num_train_epochs", type = int, default = 30)
parser.add_argument("--lr_scheduler_type", default = "linear")
parser.add_argument("--warmup_ratio", type = float, default = 0.05)
parser.add_argument("--generation_num_beams", type = int, default = 4)
parser.add_argument("--num_support_profile", type = int, default = 1)
parser.add_argument("--gradient_accumulation_steps", type = int, default = 1)
parser.add_argument("--summarizer_technique", default=None, choices=['content_only_summarization', 'stylistic_summarization', 'content_only_summarization_stylistic_generation'])
parser.add_argument("--cache_dir", default = "./cache")

if __name__ == "__main__":

    opts = parser.parse_args()
    
    model = AutoModelForSeq2SeqLM.from_pretrained(opts.model_name)
    tokenizer = AutoTokenizer.from_pretrained(opts.model_name)
    collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model = model, max_length = opts.max_length)

    task = opts.task
    if opts.use_profile:
        prompt_generator, contriver = create_prompt_generator(opts.num_support_profile, None, opts.retriever, opts.is_ranked,opts.summarizer_technique,opts.max_length, tokenizer)
    else:
        prompt_generator, contriver = None, None

    greater_is_better = True
    if task == "topic_writing":
        train_dataset = GeneralSeq2SeqDataset(opts.train_data, opts.use_profile, task, prompt_generator)
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
        test_dataset = GeneralSeq2SeqDataset(opts.test_data, opts.use_profile, task, prompt_generator)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer = tokenizer)
        best_metric = "rouge-1"
    
    elif task == "product_review_writing":
        train_dataset = GeneralSeq2SeqDataset(opts.train_data, opts.use_profile, task, prompt_generator)
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
        test_dataset = GeneralSeq2SeqDataset(opts.test_data, opts.use_profile, task, prompt_generator)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer = tokenizer)
        best_metric = "rouge-1"
    
    elif task == "generation_abstract":
        train_dataset = GeneralSeq2SeqDataset(opts.train_data, opts.use_profile, task, prompt_generator)
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
        test_dataset = GeneralSeq2SeqDataset(opts.test_data, opts.use_profile, task, prompt_generator)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer = tokenizer)
        best_metric = "rouge-1"
    
    elif task == "email_generation":
        train_dataset = GeneralSeq2SeqDataset(opts.train_data, opts.use_profile, task, prompt_generator)
        eval_dataset = GeneralSeq2SeqDataset(opts.validation_data, opts.use_profile, task, prompt_generator)
        test_dataset = GeneralSeq2SeqDataset(opts.test_data, opts.use_profile, task, prompt_generator)
        compute_metrics = create_metric_bleu_rouge_meteor(tokenizer = tokenizer)
        best_metric = "rouge-1"

    train_dataset = convert_to_hf_dataset(train_dataset).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)
    eval_dataset = convert_to_hf_dataset(eval_dataset).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)
    test_dataset = convert_to_hf_dataset(test_dataset).map(create_preprocessor(tokenizer = tokenizer, max_length = opts.max_length), batched=True)

    if contriver:
        contriver = contriver.to("cpu")

    training_args = Seq2SeqTrainingArguments(
        output_dir = opts.output_dir,
        do_train = True,
        do_eval = True,
        evaluation_strategy = "epoch",
        per_device_train_batch_size = opts.per_device_batch_size,
        per_device_eval_batch_size = 1,
        gradient_accumulation_steps = opts.gradient_accumulation_steps,
        learning_rate = opts.learning_rate,
        weight_decay = opts.weight_decay,
        num_train_epochs = opts.num_train_epochs,
        lr_scheduler_type = opts.lr_scheduler_type,
        warmup_ratio = opts.warmup_ratio,
        generation_num_beams = opts.generation_num_beams,
        predict_with_generate = True,
        save_strategy = "epoch",
        logging_steps = 50,
        eval_accumulation_steps = 1,
        generation_max_length = opts.generation_max_length,
        load_best_model_at_end = True,
        metric_for_best_model = best_metric,
        greater_is_better = greater_is_better
    )

    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        data_collator = collator,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics
    )

    trainer.train()
    print(f'experiment name is: {opts.experiment_name}')
    print(trainer.evaluate(test_dataset))
