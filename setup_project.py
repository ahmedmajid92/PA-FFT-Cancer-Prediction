import os

def create_structure():
    dirs = [
        "data/raw/mendeley_data",
        "data/raw/gse45827",
        "data/raw/tcga_pancan",
        "data/raw/msigdb",
        "data/processed",
        "src"
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")

    # Create empty __init__.py in src
    with open("src/__init__.py", "w") as f:
        pass
    print("Created src/__init__.py")

if __name__ == "__main__":
    create_structure()
