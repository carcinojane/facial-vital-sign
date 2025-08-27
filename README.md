# rPPG VS Code Starter (PURE + UBFC with Google Drive)

This project sets up a VS Code workspace to:
- Keep **PURE** and **UBFC-rPPG** datasets in **Google Drive**
- Run a fast **unsupervised POS** baseline on any mobile face video to estimate **heart rate**
- Use the official **rPPG-Toolbox** for training/evaluation via configs

## 0) Prereqs
- Miniconda/Anaconda
- Git + VS Code
- Google Drive for Desktop (or rclone mount) so your Drive appears as a normal local path
- (Optional) NVIDIA GPU with CUDA 11.8 drivers

## 1) Create env & clone toolbox
```bash
bash setup.sh
# on Windows PowerShell:
#   conda env create -f env.yml; conda activate rppg
#   git clone https://github.com/ubicomplab/rPPG-Toolbox.git rppg_toolbox
```

## 2) Put datasets in Google Drive
Create `<My Drive>/rppg_datasets/` with:
```
rppg_datasets/
├─ PURE/
│  ├─ 01-01/
│  │  ├─ 01-01/
│  │  └─ 01-01.json
│  └─ ...
└─ UBFC-rPPG/
   ├─ subject1/
   │  ├─ vid.avi
   │  └─ ground_truth.txt
   └─ subject2/...
```

## 3) Link Drive to this project
Set `GDRIVE_RPPG` if needed, then:
```bash
python scripts/make_symlinks.py
python scripts/verify_layout.py
```

## 4) Run POS baseline on a phone video
```bash
python scripts/run_unsupervised_pos.py --video ./sample_video.mp4
```
> Needs at least ~5 seconds, stable lighting, and visible face.

## 5) Use rPPG-Toolbox with configs
Open `rppg_toolbox/configs/` and pick a config for PURE or UBFC, then:
```bash
python scripts/run_toolbox_main.py --config configs/eval_configs/<your_config.yaml>
```

## Notes
- If Windows symlinks fail, enable **Developer Mode** or run terminal as **Administrator**.
- For CPU-only, remove `cudatoolkit=11.8` from `env.yml`.
- Start with unsupervised baselines to avoid heavy dependencies. Add supervised models later.
