
import os
from pathlib import Path
from training import config

target_dir = config.RAW_DATA_DIR
print(f"Target Dir: {target_dir}")
print(f"Exists: {target_dir.exists()}")
print(f"Is Dir: {target_dir.is_dir()}")

items = os.listdir(target_dir)
print(f"Items found: {len(items)}")
for item in items[:5]:
    p = target_dir / item
    print(f"Item: {item}, IsDir: {p.is_dir()}")
