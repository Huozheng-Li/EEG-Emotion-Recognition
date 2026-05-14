# EEG Emotion Recognition — Competition Project

## Task
Binary EEG emotion classification: neutral (0) vs positive (1).
- 30 EEG channels, 60 subjects (40 healthy controls + 20 depression patients)
- Training data: 50,000 timepoints per emotion per subject (estimated 250Hz)
- Test data: 10 subjects × 20,000 timepoints each
- Output format: `user_id, trial_id, Emotion_label` xlsx

## Environment
- **Windows local**: conda `11thbmedesign` in `D:\Miniconda3\envs\` (Python 3.10, PyTorch 2.11+cu126, RTX 4050 6GB)
- **Cloud AutoDL**: V100-32GB, SSH `ssh -p 54172 root@region-46.seetacloud.com`
  - Project at `/root/autodl-tmp/eeg_emotion_recog/`
  - Conda env `eeg` (Python 3.10, PyTorch 2.7+cu118)
  - Data at `/root/autodl-tmp/eeg_emotion_recog/PROBLEM/`
  - **Always use `screen -S train` for long runs** (Ctrl+A D to detach)

## Repo Structure
```
src/
  config.py                       — All paths, model params, training hyperparams
  data/
    preprocessing.py              — Bandpass filter, segmentation, standardization, DE/PSD/Hjorth extraction
    deap_dataset.py               — DEAP loader (32→30ch alignment, valence binarization)
    competition_dataset.py        — Competition loader (h5py for train, scipy for test)
  models/
    tsception.py                  — TSception (Temporal multi-scale + Asymmetric spatial + fusion)
  train/
    pretrain.py                   — DEAP pretraining script
    finetune.py                   — Competition finetuning with GroupKFold CV
    lightgbm_baseline.py          — Handcrafted features + LightGBM with GroupKFold CV
PROBLEM/                          — Competition raw data (gitignored, ~700MB zipped)
data/processed/                   — Cached preprocessed .npz files (gitignored)
checkpoints/                      — Model weights (gitignored)
```

## Key Technical Decisions

### Cross-validation: GroupKFold (subject-wise)
We do NOT use random split. Why: same subject's EEG epochs in both train and val causes data leakage — model learns subject identity, not emotion. GroupKFold ensures all epochs from a subject stay together in either train or val. Baseline went from 50% (random split) to 54% (GroupKFold) on TSception.

### Trial segmentation
- Raw: 50,000 timepoints per subject per emotion
- Segment: 5s windows, 2.5s stride (50% overlap) at 250Hz
- Each epoch: 30 channels × 1250 timepoints
- Result: ~158 epochs per subject, ~9,480 total

### Features for LightGBM
Three groups, 390 dimensions total:
1. **DE** (Differential Entropy): 5 bands × 30 channels = 150 features
   - DE = 0.5 × log(2πeσ²) per band per channel
2. **PSD** (Power Spectral Density): 5 bands × 30 channels = 150 features
   - Welch's method, log-summed band power
3. **Hjorth parameters**: 3 params × 30 channels = 90 features
   - Activity, Mobility, Complexity
- Frequency bands: δ(1-4), θ(4-8), α(8-14), β(14-31), γ(31-50) Hz

## Current Results (5-fold GroupKFold CV)

| Model | Mean Acc | Std | Train Time | Key Finding |
|-------|----------|-----|------------|--------------|
| TSception (scratch) | 54.02% | 0.84% | ~2h total | 311K params, overfits without pretraining |
| LightGBM + handcrafted | **62.19%** | 1.91% | ~4min total | **Top features**: DE_gamma(ch18,12,17), DE_alpha(ch13,16,27), Hjorth_comp(ch22) |

### Why LightGBM > TSception
- Small sample (60 subjects) + high-dim raw signal → DL overfits
- Handcrafted features inject prior knowledge (which frequency bands matter)
- LightGBM has far fewer parameters (~2000 vs 311K)
- γ and α bands confirmed as most discriminative — matches neuroscience literature

## Next Steps (Priority Order)
1. **Get SEED dataset approved** — SJTU BCMI lab, need advisor email (ask counselor tomorrow)
2. **Pretrain TSception on SEED** → finetune on competition data
3. **Model ensemble**: TSception (pretrained) + LightGBM → voting/stacking
4. **Test set inference**: segment test data → extract features / run model → fill xlsx template
5. **Paper writing**: ablation studies (with/without pretraining, with/without handcrafted features), comparison experiments

## Common Commands

### Local (Windows)
```powershell
D:\Miniconda3\envs\11thbmedesign\python -m src.train.finetune --pretrained none
```

### Cloud (AutoDL)
```bash
cd /root/autodl-tmp/eeg_emotion_recog
source activate eeg
screen -S train
python -m src.train.lightgbm_baseline   # ML pipeline, CPU, ~4min
python -m src.train.finetune --pretrained none  # DL pipeline, GPU, ~2h
# Ctrl+A D to detach, screen -r train to reattach
```

### Git workflow
```bash
# Local: make changes → commit → push
git add -A && git commit -m "message" && git push

# Cloud: pull and run
git pull
```

## Data Pipeline Cache
- `data/processed/competition_train.npz` — preprocessed epochs (9480 × 30 × 1250)
- `data/processed/competition_features.npz` — extracted 390-dim features
- Delete these to force re-processing when changing segmentation/feature params
