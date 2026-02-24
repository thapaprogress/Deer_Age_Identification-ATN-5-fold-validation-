from training import config
from utils.data_loader import create_data_loaders
from pathlib import Path
import sys

print(f"Config RAW_DATA_DIR: {config.RAW_DATA_DIR}")
if not config.RAW_DATA_DIR.exists():
    print("ERROR: RAW_DATA_DIR does not exist!")
    sys.exit(1)

print("Listing first 5 items in RAW_DATA_DIR:")
try:
    for i, p in enumerate(list(config.RAW_DATA_DIR.iterdir())[:5]):
        print(f"  {p}")
except Exception as e:
    print(f"Error listing dir: {e}")

print("\nAttempts creation of data loaders...")
try:
    loaders = create_data_loaders(
        data_dir=config.RAW_DATA_DIR,
        batch_size=8,
        num_workers=0
    )
    print("SUCCESS: Data loaders created.")
    print(f"Train batches: {len(loaders['train'])}")
except Exception as e:
    print(f"FAILURE: {e}")
