#!/usr/bin/env python
# coding: utf-8


from tqdm.auto import tqdm

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import get_scheduler

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from accelerate import Accelerator

from rouge_chinese import Rouge
import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")    

def load_data(data_path):
    print("Load data from: {}.".format(data_path))
    df = pd.read_csv(data_path) 
    df = df[['title','abstract']]
    df['abstract']=df['abstract'].astype(str)
    df['title']=df['title'].astype(str)
    print(df.head())
    return df.iloc[:50]

def split_data(df, val_size):
    print("Split data into training and validation sets, and the validation size is {}.".format(val_size))
    # Split the dataframe into train and remaining data
    train_df, remaining_df = train_test_split(df, test_size=val_size, random_state=42)

    # Split the remaining data into validation and test sets
    # validation_df, test_df = train_test_split(remaining_df, test_size=0.33, random_state=42)
    # Create the DatasetDict
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(remaining_df),
    })

    return dataset_dict


def load_model(model_checkpoint):
    print("Load '{}' model from huggingface.".format(model_checkpoint))
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    while True:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, resume_download=True)
            break
        except Exception as e:
            print(f"Error: {e}. Retrying download...")

    return tokenizer, model

def tokenize(tokenizer, dataset_dict, max_input_length, max_target_length):
    print("Tokenize the input data")
    def _preprocess(examples):

        model_inputs = tokenizer(
            examples["abstract"],
            max_length=max_input_length,
            truncation=True,
            
        )
        labels = tokenizer(
            examples["title"], 
            max_length=max_target_length, 
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_datasets = dataset_dict.map(_preprocess, batched=True)
    return tokenized_datasets

#將生成的摘要拆分為由換行符分隔的句子。這是 ROUGE 指標期望的格式，
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels
    
def training(model, train_dataloader, eval_dataloader, num_train_epochs, lr, tokenizer, output_dir):
    optimizer = AdamW(model.parameters(), lr=lr)
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
    progress_bar = tqdm(range(num_training_steps))

    epoch_results = []  # Initialize list to store Rouge scores for each epoch
    epoch_scores_bleu = [] # Initialize list to store bleu scores for each epoch

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        scores = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]

                # If we did not pad to max length, we need to pad the labels too
                labels = accelerator.pad_across_processes(
                    batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
                )

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                # Replace -100 in the labels as we can't decode them
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(
                    decoded_preds, decoded_labels
                )
                # Convert elements to strings
                decoded_labels_str = ' '.join(decoded_labels[0])
                decoded_preds_str = decoded_preds[0].replace('<extra_id_0>','')
                decoded_preds_str = ' '.join(decoded_preds_str) if decoded_preds_str else '_'



                rouge = Rouge()
                result = rouge.get_scores(decoded_preds_str, decoded_labels_str)[0]
                # Calculate mean scores for each epoch
                epoch_results.append(result)  # Append Rouge scores to epoch_results list

                # Calculate BLEU score
                references = [decoded_labels_str.split()]
                hypothesis = decoded_preds_str.split()
                score = nltk.translate.bleu_score.sentence_bleu(references, hypothesis)
                scores.append(score)
        mean_scores = {
            'rouge-1': {
                'r': np.mean([score['rouge-1']['r'] for score in epoch_results]),
                'p': np.mean([score['rouge-1']['p'] for score in epoch_results]),
                'f': np.mean([score['rouge-1']['f'] for score in epoch_results])
            },
            'rouge-2': {
                'r': np.mean([score['rouge-2']['r'] for score in epoch_results]),
                'p': np.mean([score['rouge-2']['p'] for score in epoch_results]),
                'f': np.mean([score['rouge-2']['f'] for score in epoch_results])
            },
            'rouge-l': {
                'r': np.mean([score['rouge-l']['r'] for score in epoch_results]),
                'p': np.mean([score['rouge-l']['p'] for score in epoch_results]),
                'f': np.mean([score['rouge-l']['f'] for score in epoch_results])
            }
        }

        print(f"Epoch {epoch}:",f"Mean Scores: {mean_scores}")  # Print mean scores for each epoch
        # Calculate mean scores for each epoch
        epoch_score_bleu = sum(scores) / len(scores)
        epoch_scores_bleu.append(epoch_score_bleu)
        print(f"Epoch {epoch}: Mean BLEU score = {epoch_score_bleu}")

        # Save and upload
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
    
    
def run(training_data_path, val_size, model_checkpoint, max_input_length, max_target_length, batch_size, num_train_epochs, learning_rate, output_dir):
    df = load_data(data_path=training_data_path)
    dataset_dict = split_data(df=df, val_size=val_size)
    
    tokenizer, model = load_model(model_checkpoint=model_checkpoint)
    
    tokenized_datasets = tokenize(tokenizer, dataset_dict, max_input_length, max_target_length)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    tokenized_datasets = tokenized_datasets.remove_columns(
        dataset_dict["train"].column_names
    )
    data_collator([tokenized_datasets["train"][i] for i in range(2)])
    tokenized_datasets.set_format("torch")
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], collate_fn=data_collator, batch_size=batch_size
    )
    
    training(model, train_dataloader, eval_dataloader, num_train_epochs, learning_rate, tokenizer, output_dir)

if __name__ == "__main__":
    run(training_data_path='training_data.csv', 
        val_size=0.3,
        model_checkpoint="google/mt5-small",
        max_input_length=1024, #設置摘要的長度上限
        max_target_length=50, #設置標題的長度上限
        batch_size=1,
        num_train_epochs=10,
        output_dir = "results-mt5-finetuned-squad-accelerate_v1",
        learning_rate=2e-5
    )
    

        


