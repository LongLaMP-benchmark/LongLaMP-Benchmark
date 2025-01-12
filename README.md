# Long text Large Language Model Personalization (longLaMP)

Large language models (LLMs) have demonstrated remarkable capabilities in text generation tasks, but their performance in personalized long-text generation remains largely unexplored. While recent works have focused on personalization for short-text generation, there is a need for a comprehensive evaluation framework tailored to long-text generation, considering its widespread real-world applications.

LongLaMP aims to fill this gap by providing a benchmark specifically designed to assess the effectiveness of LLMs in generating personalized, long-text outputs. It is a collection of datasets spanning diverse domains, each focused on the challenging task of long-text generation.

If you use the LongLaMP benchmark in your work, please cite: [LongLaMP: Personalized Long-text Generation](https://www.arxiv.org/abs/2407.11016). Thanks!

### Data

You can download all the datasets from the links provided here:

[LongLaMP Data](https://huggingface.co/datasets/LongLaMP/LongLaMP)

Personalized Email Subject Generation (Avocado dataset):

The Avocado dataset is not publicly accessible. However, we provided the samples' id and the code we used to generate our dataset. Therefore, if you get access to the dataset, you can quickly generate the dataset with the same format as the other datasets in LaMP using the following code:

`python data/avocado/create_avocado_dataset.py \
    --avocado_files_dir \*Address to the directory containing zip files for avocado dataset 'avocado-1.0.2/data/text'*\ \
    --extract_addr \*A temp dir to extract the files for creating dataset*\ \
    --output_dir \*The directory to generate the final dataset*\ \
    --input_question_file_train \*The address to the train_questions.json file we provided in LaMP*\ \
    --input_question_file_dev \*The address to the dev_questions.json file we provided in LaMP*\ \
    --input_question_file_test \*The address to the test_questions.json file we provided in LaMP*\`

### Evaluation
The instructions for evaluating your results on the test set are provided here. To evaluate your results on the dev set, we provided an evaluation script that can be found here:

Evaluate all tasks together:

`python eval/eval_all.py \
    --golds_zip /*Address to all gold labels for all tasks zipped in a file*/ \
    --preds_zip /*Address to all predictions for all tasks zipped in a file*/ \
    --temp_dir /*Address to a temp dir for extracting files*/ \
    --output_file /*Address to the results file*/ \`

Evaluate one task:

`python eval/eval_task.py \
    --golds_json /*Address to gold labels for the task as a json file*/ \
    --preds_json /*Address to predictions for the task as a json file*/ \
    --task_name 
    --output_file /*Address to the results file*/ \`

The pred files should follow the exact same format as the gold files:

`{
    "task" : "/*task name*/",
    "golds" : [
        {
            "id" : "/*sample 1 id*/",
            "output" : "/*output of the model for the first sample*/"
        },
        ...,
        {
            "id" : "/*sample n id*/",
            "output" : "/*output of the model for the n'th sample*/"
        }
    ]
}`

### Training LLM with RAG
The next step is to train the LLM on a LaMP task:

`cd LongLaMP
python train_llm.py \
    --train_data /*address to sorted training data using the previous step*/ \
    --validation_data /*address to sorted validation data using the previous step*/ \
    [optional] --test_data /*address to sorted test data using the previous step*/ \
    --model_name /*address to the model that should be used for initialization of the LLM*/ \
    --task \
    --output_dir /*output directory to save results and checkpoints*/ \
    --retriever /*the ranking model to be used [bm25, contriever, recency]*/ \
    --use_profile \ /*used to perfrom personalization with RAG */
    --is_ranked \ /*used if you pre-ranked the profiles based on the provided retrieval model*/
    --num_retrieved /*number of items to be retrieved from the user profile*/ \ `

### Zero-shot Evaluation of LLM with RAG
You can also evaluate the LLMs with the following script:

`cd LongLaMP
python evaluate_llm.py \
    --validation_data /*address to sorted validation data using the previous step*/ \
    --model_addr /*address to the model that should be used for initialization of the LLM*/ \
    --task\
    --output_dir /*output directory to save results */ \
    --use_profile \ /*used to perfrom personalization with RAG */
    --retriever /*the ranking model to be used [bm25, contriever, recency]*/ \
    --is_ranked \ /*used if you pre-ranked the profiles based on the provided retrieval model*/
    --num_retrieved /*number of items to be retrieved from the user profile*/ \ `

