hf_dataset_identifier = "segments/sidewalk-semantic"

from datasets import load_dataset


ds = load_dataset(hf_dataset_identifier)



ds = ds.shuffle(seed=1)
ds = ds["train"].train_test_split(test_size=0.2)
train_ds = ds["train"]
test_ds = ds["test"]

breakpoint()