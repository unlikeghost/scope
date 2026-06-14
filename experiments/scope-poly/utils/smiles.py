try:
    from rdkit import Chem
except ImportError:
    raise ImportError(
        "Please install the sklearn package"
        "If you are using uv run uv add uv sync --group experiments"
    )

def standardize_dataset(
    smiles: str,
) -> str | None:
    return smiles