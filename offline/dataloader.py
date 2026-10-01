# from datasets import load_dataset_builder, load_dataset, get_dataset_config_names

# MAX_ROWS = 100

# # ds_builder = load_dataset_builder("allenai/ai2_arc", "ARC-Easy")

# # print(ds_builder.info.features)

# ds = load_dataset("allenai/ai2_arc", "ARC-Easy")

# shuffled_dataset = ds.shuffle(seed=42)

# train_ds = shuffled_dataset["train"].select([i for i in range(100)])
# test_ds = shuffled_dataset["test"].select([i for i in range(100)])
# validation_ds = shuffled_dataset["validation"].select([i for i in range(50)])

# # print(train_ds)
# # print(test_ds)
# # print(validation_ds)

# # configs = get_dataset_config_names("allenai/ai2_arc")
# # print(configs)

# # print(train_ds)


# one_line = train_ds[0]
# question = one_line["question"]
# choices = one_line["choices"]
# answerKey = one_line["answerKey"]
# answerIndex = choices["label"].index(answerKey)
# answer = choices["text"][answerIndex]

# print(one_line)
# print(question)
# print(choices)
# print(answerKey)
# print(answerIndex)
# print(answer)





# from datasets import load_dataset

# MAX_ROWS = 50

# # MMLU-Pro has two official splits:
# # - validation: 70 examples
# # - test: 12032 examples
# ds = load_dataset(
#     "TIGER-Lab/MMLU-Pro",
#     split="test"
# )

# # Shuffle to avoid taking only the first questions
# shuffled_dataset = ds.shuffle(seed=42)

# # For now we only take 50 examples to test the model gap
# train_ds = shuffled_dataset.select(range(MAX_ROWS))


# if __name__ == "__main__":
#     print(train_ds)

#     one_line = train_ds[0]

#     question = one_line["question"]
#     options = one_line["options"]
#     answer = one_line["answer"]
#     answer_index = one_line["answer_index"]
#     category = one_line["category"]

#     print("\nQuestion:")
#     print(question)

#     print("\nOptions:")
#     for i, option in enumerate(options):
#         label = chr(ord("A") + i)
#         print(f"{label}: {option}")

#     print("\nCorrect answer:")
#     print(answer)

#     print("\nAnswer index:")
#     print(answer_index)

#     print("\nCategory:")
#     print(category)



from datasets import load_dataset


MAX_ROWS = 50


ds = load_dataset(
    "TIGER-Lab/MMLU-Pro",
    split="test"
)

shuffled_dataset = ds.shuffle(seed=42)

train_ds = shuffled_dataset.select(
    range(MAX_ROWS)
)


if __name__ == "__main__":

    print(train_ds)

    one_line = train_ds[0]

    question = one_line["question"]
    options = one_line["options"]
    answer = one_line["answer"]
    answer_index = one_line["answer_index"]
    category = one_line["category"]

    print("\nQuestion:")
    print(question)

    print("\nOptions:")

    for i, option in enumerate(options):
        label = chr(ord("A") + i)
        print(f"{label}: {option}")

    print("\nCorrect answer:")
    print(answer)

    print("\nAnswer index:")
    print(answer_index)

    print("\nCategory:")
    print(category)