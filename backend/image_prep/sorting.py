import shutil
from pathlib import Path

here = Path(__file__).resolve().parent.parent / "data"
sort_dir = here / "to_sort"
training_dir = here / "training"

print("sort dir: ", sort_dir)
print("training dir: ", training_dir)

categories = ["0_mouse_bite", "5_spurious_copper", "1_spur", "2_missing_hole", "3_short", "4_open_circuit"]


def safe_move_to_category(src_path: Path, category: str):
    dst_dir = training_dir / category
    dst_dir.mkdir(parents=True, exist_ok=True)

    candidates = [src_path]

    for candidate in candidates:
        if candidate.exists():
            try:
                shutil.move(str(candidate), str(dst_dir))
                print(f"Moved: \n{candidate} \n-> \n{dst_dir}\n")
                return True
            except Exception as e:
                print(f"Error moving {candidate} -> {dst_dir}: {e}")
                return False

    print(f"No candidate found for: {src_path} (tried: {[p.name for p in candidates]})")
    return False


if not sort_dir.exists():
    print("Sort directory does not exist:", sort_dir)
else:
    for entry in sort_dir.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        for cat in categories:
            keyword = cat[2:]
            if keyword in name:
                safe_move_to_category(entry, cat)
                break
