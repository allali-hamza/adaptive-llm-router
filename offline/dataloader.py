from datasets import load_dataset_builder, load_dataset, get_dataset_config_names

MAX_ROWS = 100

# ds_builder = load_dataset_builder("allenai/ai2_arc", "ARC-Easy")

# print(ds_builder.info.features)

ds = load_dataset("allenai/ai2_arc", "ARC-Easy")

shuffled_dataset = ds.shuffle(seed=42)

train_ds = shuffled_dataset["train"].select([i for i in range(100)])
test_ds = shuffled_dataset["test"].select([i for i in range(100)])
validation_ds = shuffled_dataset["validation"].select([i for i in range(50)])

# print(train_ds)
# print(test_ds)
# print(validation_ds)

# configs = get_dataset_config_names("allenai/ai2_arc")
# print(configs)

# print(train_ds)


one_line = train_ds[0]
question = one_line["question"]
choices = one_line["choices"]
answerKey = one_line["answerKey"]
answerIndex = choices["label"].index(answerKey)
answer = choices["text"][answerIndex]

print(one_line)
print(question)
print(choices)
print(answerKey)
print(answerIndex)
print(answer)