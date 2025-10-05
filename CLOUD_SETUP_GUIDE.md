# Cloud Setup Guide for PhysNet Fine-tuning

## Option 1: Google Colab (Recommended - Free GPU)

### Advantages
- **Free GPU**: T4 GPU with 15GB VRAM
- **No setup**: Ready to use immediately
- **Easy sharing**: Collaborative notebooks

### Steps

1. **Create Colab Notebook**
   - Go to https://colab.research.google.com
   - Click "New Notebook"

2. **Enable GPU**
   - Runtime → Change runtime type → Hardware accelerator → GPU → Save

3. **Upload Dataset**
   ```python
   # Option A: Upload from local (slow for large files)
   from google.colab import files
   uploaded = files.upload()

   # Option B: Mount Google Drive (recommended)
   from google.colab import drive
   drive.mount('/content/drive')

   # Option C: Download from cloud storage
   !wget YOUR_DATASET_URL
   ```

4. **Install rPPG Toolbox**
   ```python
   # Clone the repository
   !git clone https://github.com/ubicomplab/rPPG-Toolbox.git
   %cd rPPG-Toolbox

   # Install dependencies
   !pip install -r requirements.txt
   ```

5. **Upload Your Files**
   ```python
   # Upload augmented dataset (UBFC_MOBILE)
   # Upload physnet_mobile_finetune.yaml
   # Upload augmentation scripts if needed
   ```

6. **Run Training**
   ```python
   # Verify GPU is available
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU: {torch.cuda.get_device_name(0)}")

   # Run training
   !python main.py --config_file physnet_mobile_finetune.yaml
   ```

### Complete Colab Notebook Example

```python
# ========================================
# PhysNet Mobile Fine-tuning - Google Colab
# ========================================

# 1. Setup GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 2. Mount Google Drive (upload your dataset here first)
from google.colab import drive
drive.mount('/content/drive')

# 3. Clone rPPG-Toolbox
!git clone https://github.com/ubicomplab/rPPG-Toolbox.git
%cd rPPG-Toolbox

# 4. Install dependencies
!pip install -r requirements.txt

# 5. Copy dataset from Google Drive to Colab workspace
!cp -r "/content/drive/MyDrive/UBFC_MOBILE" /content/data/UBFC_MOBILE

# 6. Create config file
config_content = """
BASE: ['']
TOOLBOX_MODE: "train_and_test"
DEVICE: cuda:0
NUM_OF_GPU_TRAIN: 1

TRAIN:
  BATCH_SIZE: 8  # GPU can handle larger batches
  EPOCHS: 30
  LR: 0.00001
  MODEL_FILE_NAME: physnet_mobile_finetuned
  PLOT_LOSSES_AND_LR: True
  DATA:
    FS: 30
    DATASET: UBFC-rPPG
    DO_PREPROCESS: True
    DATA_FORMAT: NCDHW
    DATA_PATH: "/content/data/UBFC_MOBILE"
    CACHED_PATH: "/content/cache_mobile"
    BEGIN: 0.0
    END: 0.7
    PREPROCESS:
      DATA_TYPE: ['DiffNormalized']
      LABEL_TYPE: DiffNormalized
      DO_CHUNK: True
      CHUNK_LENGTH: 128
      CROP_FACE:
        DO_CROP_FACE: True
        BACKEND: 'HC'
        USE_LARGE_FACE_BOX: True
        LARGE_BOX_COEF: 1.5
      RESIZE:
        H: 72
        W: 72

VALID:
  DATA:
    FS: 30
    DATASET: UBFC-rPPG
    DO_PREPROCESS: True
    DATA_PATH: "/content/data/UBFC_MOBILE"
    CACHED_PATH: "/content/cache_mobile"
    BEGIN: 0.7
    END: 0.85

TEST:
  METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR']
  DATA:
    FS: 30
    DATASET: UBFC-rPPG
    DO_PREPROCESS: True
    DATA_PATH: "/content/data/UBFC_MOBILE"
    CACHED_PATH: "/content/cache_mobile"
    BEGIN: 0.85
    END: 1.0

MODEL:
  DROP_RATE: 0.2
  NAME: Physnet
  MODEL_DIR: /content/runs/mobile_finetune
  PHYSNET:
    FRAME_NUM: 128
  RESUME: "final_model_release/PURE_PhysNet_DiffNormalized.pth"

LOG:
  PATH: /content/runs/mobile_finetune

INFERENCE:
  BATCH_SIZE: 8
  EVALUATION_METHOD: FFT
"""

with open('physnet_mobile_finetune.yaml', 'w') as f:
    f.write(config_content)

# 7. Start training
!python main.py --config_file physnet_mobile_finetune.yaml

# 8. Download trained model
from google.colab import files
files.download('/content/runs/mobile_finetune/physnet_mobile_finetuned_best.pth')
```

---

## Option 2: Kaggle (Alternative Free GPU)

### Advantages
- Free Tesla P100 GPU (16GB VRAM)
- 30 hours/week GPU quota
- Direct dataset upload

### Steps

1. Go to https://www.kaggle.com
2. Click "Create" → "New Notebook"
3. Settings → Accelerator → GPU
4. Upload dataset as Kaggle Dataset
5. Similar code to Colab above

---

## Option 3: AWS/Azure/GCP (Paid but Powerful)

### AWS SageMaker
```bash
# Estimated cost: $1-3 per hour
# Instance type: ml.p3.2xlarge (V100 GPU)

# 1. Create SageMaker notebook instance
# 2. Upload dataset to S3
# 3. Clone repository
# 4. Run training
```

### Google Cloud AI Platform
```bash
# Estimated cost: $1.50 per hour
# Machine type: n1-standard-4 with 1 x NVIDIA Tesla T4

gcloud ai-platform jobs submit training physnet_mobile_$(date +%s) \
  --region=us-central1 \
  --master-image-uri=gcr.io/deeplearning-platform-release/pytorch-gpu \
  --scale-tier=BASIC_GPU \
  --package-path=./rppg_toolbox \
  --module-name=rppg_toolbox.main \
  -- \
  --config_file=physnet_mobile_finetune.yaml
```

---

## Option 4: University Computing Resources

Check if NUS provides:
- GPU compute clusters
- Research computing services
- Cloud credits (AWS Educate, Azure for Students)

Contact: NUS IT / Your department's computing support

---

## Recommended Workflow

### For Your Project (Best Approach):

1. **Upload augmented dataset to Google Drive**
   - Compress UBFC_MOBILE folder
   - Upload to Google Drive (takes 10-20 mins)

2. **Use Google Colab**
   - Free T4 GPU
   - Training will take 1-2 hours instead of 8+ hours on CPU
   - No memory issues with GPU

3. **Download trained model**
   - Save to Google Drive
   - Download to local machine for deployment

### Quick Start (5 minutes to begin training):

1. Compress your dataset:
   ```bash
   cd "C:\Users\janej\OneDrive - National University of Singapore\Capstone Project\rppg-vscode-starter"
   tar -czf UBFC_MOBILE.tar.gz data/UBFC_MOBILE
   ```

2. Upload to Google Drive manually or:
   ```bash
   # Install rclone (optional)
   rclone copy UBFC_MOBILE.tar.gz gdrive:rppg_project/
   ```

3. Open https://colab.research.google.com/

4. Copy the complete notebook code above

5. Run all cells

---

## Storage Requirements

- **Local**: 4-6 GB (augmented + original dataset)
- **Cloud**: 2-3 GB (compressed augmented dataset only)
- **Colab workspace**: 15 GB free temporary storage
- **Google Drive**: 15 GB free (enough for this project)

---

## Training Time Comparison

| Environment | Hardware | Estimated Time | Cost |
|-------------|----------|----------------|------|
| Local CPU | Your laptop | 8-12 hours | Free |
| Colab GPU | Tesla T4 | 1-2 hours | Free |
| Kaggle GPU | Tesla P100 | 1-1.5 hours | Free |
| AWS GPU | Tesla V100 | 30-45 mins | $1-3 |

---

## Next Steps

1. **Immediate**: Use Google Colab (see notebook code above)
2. **Compress and upload dataset** to Google Drive
3. **Run notebook** with GPU enabled
4. **Download trained model** when complete
5. **Use locally** for inference/deployment

Would you like me to create a ready-to-use Colab notebook file for you?
