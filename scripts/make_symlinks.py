import os, pathlib, sys

# Change this env var to your Google Drive dataset root if needed
GDRIVE_ROOT = os.environ.get("GDRIVE_RPPG", r"G:\My Drive\iss\Capstone_Project\Vital_sign_scan_pretrained\data")
PROJECT_DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
PROJECT_DATA.mkdir(exist_ok=True)

links = {
    "PURE": pathlib.Path(GDRIVE_ROOT) / "PURE",
    "UBFC-rPPG": pathlib.Path(GDRIVE_ROOT) / "UBFC-rPPG",
}

def make_symlink(src: pathlib.Path, dst: pathlib.Path):
    if dst.exists() or dst.is_symlink():
        try:
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                # Remove empty dir if present
                dst.rmdir()
        except Exception:
            pass
    try:
        dst.symlink_to(src, target_is_directory=True)
        print(f"Linked {dst} -> {src}")
    except PermissionError as e:
        print("❗ Symlink permission error. On Windows, enable Developer Mode or run terminal as Admin.")
        print("   Alternatively, set RPPG_DATA_ROOT to your Google Drive path and skip symlinks.")
        raise e

if __name__ == "__main__":
    for name, target in links.items():
        if not target.exists():
            print(f"Missing: {target}. Create it in Google Drive first."); sys.exit(1)
        make_symlink(target, PROJECT_DATA / name)
    print("✅ Symlinks created at", PROJECT_DATA)
    print("   Set RPPG_DATA_ROOT to this path if needed.")
