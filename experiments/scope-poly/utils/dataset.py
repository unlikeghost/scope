import numpy as np
from typing import (List, Union)

try:
    import pandas as pd
    from collections import Counter
    from sklearn.model_selection import train_test_split
except ImportError:
    raise ImportError(
        "Please install the sklearn package"
        "If you are using uv run uv add uv sync --group experiments"
    )

from scope import SampleGenerator

from .smiles import standardize_dataset


def validate_dataset(xs: list, kws: list, name: str):
    print(f"\nValidating {name} dataset...")
    assert len(xs) == len(set(xs)), f"{name}: queries duplicated"
    for x, kw in zip(xs, kws):
        for supports in kw.values():
            assert x not in supports, f"{name}: query is in support values"
    print(f"{name} dataset is valid.")
    
    
def load_dataset(
    dataset_path: str,
    smiles_column: str,
    label_columns: Union[str, List[str]],
    min_length: int = 1,
    search_size: float = 0.2,
    random_seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    data = pd.read_csv(dataset_path)
    data["smiles_std"] = data[smiles_column].apply(standardize_dataset)
    
    label_columns = [label_columns] if isinstance(label_columns, str) else label_columns
    cols_to_drop = set(label_columns) | {smiles_column}

    filtered_dataset = (
        data
        .dropna(subset=["smiles_std"])
        .loc[lambda df: df["smiles_std"].str.len() >= min_length]
        .drop_duplicates(subset=["smiles_std"], keep="first")
        .reset_index(drop=True)
    )
   
    filtered_dataset["target"] = (
        filtered_dataset
        .groupby(label_columns, sort=False)
        .ngroup()
    )
    
    groups = filtered_dataset.groupby(label_columns, sort=False)
    filtered_dataset["target"] = groups.ngroup()
    
    mapping_df = (
        filtered_dataset
        .drop_duplicates("target")
        .set_index("target")[label_columns]
    )

    label_map = mapping_df.to_dict(orient="index")

    filtered_dataset = filtered_dataset.drop(
        columns=cols_to_drop
    )
    
    labels = filtered_dataset["target"]
    
    search, test = train_test_split(
        filtered_dataset,
        test_size=search_size,
        random_state=random_seed,
        stratify=labels
    )
    
    print("\nDataset Summary")
    print("\nLabel Mapping Reference:")
    for target_id, original_vals in label_map.items():
        print(f"ID {target_id} -> {original_vals}")
            
    print(f"Total samples in the search set: {len(search)}")
    print(f"Total samples in the evaluation set: {len(test)}")
    print(f"Total count per samples per class in the search set: {Counter(search['target'])}")
    print(f"Total count per class in the evaluation set: {Counter(test['target'])}")
    
    return search, test


def build_dataset_by_sample(
    dataframe: pd.DataFrame,
    sample_sizes: list,
    random_seed: int = 42
) -> tuple[dict, dict, dict]:
    
    x_, y_ = dataframe["smiles_std"].values, dataframe["target"].values
    
    gen = SampleGenerator(
        x=x_,
        y=y_,
        seed=random_seed
    )
    
    all_x = {sample: [] for sample in sample_sizes}
    all_y = {sample: [] for sample in sample_sizes}
    all_kw = {sample: [] for sample in sample_sizes}

    for sample in sample_sizes:
        for index, x_i, y_i, kw_i in gen.sampling(num_samples=sample):
            all_x[sample].append(x_i)
            all_y[sample].append(y_i)
            all_kw[sample].append(kw_i)

    print("\nSampling Dataset Summary")
    for s in sample_sizes:
        n_samples = len(all_x[s])
        class_dist = Counter([int(y) for y in all_y[s]])
        print(f"Size {s:2d}: {n_samples:4d} samples | Class Dist: {dict(class_dist)}")

    return all_x, all_y, all_kw
    

def build_dataset_variable(
    dataframe: pd.DataFrame,
    sample_sizes: list,
    random_seed: int = 42
) -> tuple[list, list, list]:
    x_, y_ = dataframe["smiles_std"].values, dataframe["target"].values

    all_x, all_y, all_kw = [], [], []
    rng = np.random.default_rng(random_seed)

    n_total = x_.shape[0]
    possible_sizes = np.arange(min(sample_sizes), max(sample_sizes) + 1)

    sampling_sizes = np.tile(possible_sizes, int(np.ceil(n_total / len(possible_sizes))))[:n_total]
    rng.shuffle(sampling_sizes)

    generator = SampleGenerator(
        seed=random_seed,
        x=x_,
        y=y_,
    )

    for idx, support_size in enumerate(sampling_sizes):
        generator.sampling(num_samples=int(support_size))
        x_e, y_e, kw = generator[idx]
        all_x.append(x_e)
        all_y.append(y_e)
        all_kw.append(kw)

    indices = rng.permutation(len(all_x))
    all_x = [all_x[i] for i in indices]
    all_y = [all_y[i] for i in indices]
    all_kw = [all_kw[i] for i in indices]

    print("\nVariable Dataset Summary")
    print(f"Total samples: {len(all_x)}")
    print(f"Count per class: {Counter([int(y) for y in all_y])}")
    print(f"Sizes distribution: {Counter([len(list(kw.values())[0]) for kw in all_kw])}")

    validate_dataset(
        xs=all_x,
        kws=all_kw,
        name='variable'
    )

    return all_x, all_y, all_kw