import os, pathlib

DATA = pathlib.Path(os.environ.get("RPPG_DATA_ROOT", "data"))

def check_pure():
    pure = DATA / "PURE"
    assert pure.exists(), "PURE not found at data/PURE"
    samples = list(pure.glob("*/*/*.json"))
    assert samples, "No PURE .json files found under data/PURE/*/*"
    print(f"PURE OK. Found {len(samples)} jsons")

def check_ubfc():
    ubfc = DATA / "UBFC-rPPG"
    assert ubfc.exists(), "UBFC-rPPG not found at data/UBFC-rPPG"
    subs = list(ubfc.glob("subject*/ground_truth.txt"))
    assert subs, "No UBFC ground_truth.txt found under data/UBFC-rPPG/subject*"
    print(f"UBFC OK. Found {len(subs)} subjects with ground_truth.txt")

if __name__ == "__main__":
    print("Checking", DATA.resolve())
    check_pure()
    check_ubfc()
    print("✅ Layout looks good.")
