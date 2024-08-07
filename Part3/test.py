import json
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, DefaultDataCollator, pipeline
import evaluate, torch

squad = load_dataset('json', data_files='dataset.json')

squad = squad['train'].select(range(26))  

squad = squad.train_test_split(train_size=6)

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)

data_collator = DefaultDataCollator()

model = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")

training_args = TrainingArguments(
    output_dir="my_awesome_qa_model",
    eval_strategy="steps",
    eval_steps=2,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    push_to_hub=False,
)

# Load metric
metric = evaluate.load("squad")

def compute_metrics(p):
    predictions, references = p

    # Extracting start and end logits
    start_logits, end_logits = predictions
    start_positions, end_positions = references

    # Convert logits to predicted positions (argmax) and then to lists of lists of integers
    start_predictions = torch.argmax(torch.tensor(start_logits), dim=-1).tolist()
    end_predictions = torch.argmax(torch.tensor(end_logits), dim=-1).tolist()
    start_positions = start_positions.tolist()
    end_positions = end_positions.tolist()

    # Combine start and end predictions
    combined_predictions = [list(a) for a in zip(start_predictions, end_predictions)]
    combined_references = [list(a) for a in zip(start_positions, end_positions)]


    # Ensure predictions are in the correct format for batch_decode
    if isinstance(combined_predictions, list) and all(isinstance(p, list) for p in combined_predictions):
        predictions = tokenizer.batch_decode(combined_predictions, skip_special_tokens=True)
    else:
        raise ValueError("Predictions are not in the correct format")

    if isinstance(combined_references, list) and all(isinstance(r, list) for r in combined_references):
        references = tokenizer.batch_decode(combined_references, skip_special_tokens=True)
    else:
        raise ValueError("References are not in the correct format")

    # Convert predictions and references to the format needed for squad metric
    formatted_predictions = [{"id": str(i), "prediction_text": pred} for i, pred in enumerate(predictions)]
    formatted_references = [{"id": str(i), "answers": {"text": [ref], "answer_start": [0]}} for i, ref in enumerate(references)]
    
    return metric.compute(predictions=formatted_predictions, references=formatted_references)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()

with open('dataset.json', 'r') as f:
    dataset = json.load(f)

def select_title(titles):
    print("Select a title by entering the corresponding number:")
    for i, title in enumerate(titles):
        print(f"{i + 1}. {title}")
    choice = int(input("\nEnter the number of the title you choose: ")) - 1
    return choice

def get_user_question():
    return input("\nEnter your question (or type 'exit' or 'e' to quit, 'back' or 'b' to select a different title): ")

titles = [entry['title'] for entry in dataset]
contexts = [entry['context'] for entry in dataset]

while True:
    selected_index = select_title(titles)
    selected_title = titles[selected_index]
    selected_context = contexts[selected_index]

    print(f"\nYou selected: {selected_title}")
    print(f"Context: {selected_context}\n")

    while True:
        user_question = get_user_question()

        if user_question.lower() in ["exit", "e"]:
            print("Exiting the program.")
            exit()
        elif user_question.lower() in ["back", "b"]:
            print("Going back to the list of titles.")
            break

        # Loading model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForQuestionAnswering.from_pretrained("my_awesome_qa_model")

        question_answerer = pipeline("question-answering", model=model, tokenizer=tokenizer)

        answer = question_answerer(question=user_question, context=selected_context)
        print("\nAnswer:", answer['answer'])
