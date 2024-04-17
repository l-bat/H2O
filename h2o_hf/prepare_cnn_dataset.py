import datasets
import json

dictionary = {"logprobs": 1, "max_tokens": 64, "n": 1, "stop": ["###"], "temperature": 0.3, "top_p": 1}

dataset = datasets.load_dataset("EdinburghNLP/xsum", split='validation', streaming=False)
# dataset = datasets.load_dataset('cnn_dailymail', "3.0.0", split='validation', streaming=False)
for data in dataset:
    # dictionary["article"] = "###\nArticle: " + str(data["article"]) + "\n\nSummarize the above article.\n"
    dictionary["article"] = "###\nArticle: " + str(data["document"]) + "\n\nSummarize the above article in 1 sentence.\n"
    dictionary["summary_gt"] = data["summary"]
    with open("data/summarization_data/xsum_val_full.json", "a") as outfile:
        json.dump(dictionary, outfile)
        outfile.write('\n')


# dataset = datasets.load_dataset('cnn_dailymail', "3.0.0", split='train')
# # dataset["len"] = len(dataset["article"])
# dataset = dataset.add_column(name="len", column=[-len(dataset[i]["article"]) for i in range(len(dataset))])
# sorted_dataset = dataset.sort("len")
# for i, data in enumerate(sorted_dataset):
#     dictionary["article"] = "###\nArticle: " + str(data["article"]) + "\n\nSummarize the above article.\n"
#     dictionary["summary_gt"] = data["highlights"]
#     with open("/home/ltalamanova/old/H2O/h2o_hf/data/summarization_data/cnn_longest_1000.json", "a") as outfile:
#         json.dump(dictionary, outfile)
#         outfile.write('\n')
#     if i == 999:
#         break