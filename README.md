# Rhythm-DPSNN: Neural Oscillation Modulated Spiking Neural Network for Low-Latency Streaming Speech Enhancement

This repository extends the Dual-Path Spiking Neural Network (DPSNN) for low-latency streaming speech enhancement by integrating neural oscillation based rhythm modulation into selected spiking neuron layers.

The original DPSNN is a time-domain spiking neural network for speech enhancement. It avoids the latency introduced by STFT and iSTFT based frequency-domain processing by using a learned convolutional encoder and decoder. The separator follows a dual-path structure: Spiking Convolutional Neural Networks (SCNNs) capture temporal contextual information while Spiking Recurrent Neural Networks (SRNNs) model frequency-related features. This makes DPSNN suitable for low-latency applications such as hearing aids, earbuds and real-time communication systems.

This project investigates whether rhythm-based spiking neurons can improve the temporal processing and energy behavior of DPSNN. Inspired by neural oscillations, rhythm modulation periodically switches neurons between ON and OFF states. During ON states, neurons follow their normal spiking dynamics. During OFF states, spike generation is suppressed and selected internal states may be frozen depending on the configuration. The goal is to study whether structured temporal sparsity can reduce activity while preserving or improving speech enhancement quality.

## Project Overview

This repository contains:

- The original DPSNN baseline.
- Rhythm-DPSNN variants using Rhythm-ALIF neurons in the SRNN separator block.
- Rhythm-DPSNN variants using Rhythm-PLIF neurons in the SCNN separator block.
- Training and testing scripts for VoiceBank+VCTK.
- Evaluation utilities for speech quality and efficiency metrics.
- Energy proxy evaluation based on SynOPS, spike activity, PDP and estimated Joules.
- Audio demos for qualitative comparison.

## Main Research Question

Can neural oscillation based rhythmic modulation improve the efficiency, robustness or perceptual quality of a low-latency spiking speech enhancement model?

More specifically, this project compares:

1. Plain DPSNN baseline.
2. Rhythm-DPSNN with rhythm modulation in the SRNN block.
3. Rhythm-DPSNN with rhythm modulation in the SCNN block.
4. Fixed rhythm assignment versus alternative rhythm configurations.

## Background

### DPSNN

DPSNN is a spiking neural network for low-latency streaming speech enhancement. It follows an encoder-separator-decoder structure:

1. **Encoder**  
   Converts raw waveform frames into learned feature maps using convolutional filters. This replaces STFT and reduces algorithmic latency.

2. **Separator**  
   Estimates a mask over the encoded representation. The separator contains:
   - SCNN modules for temporal context modeling.
   - SRNN modules for frequency-related recurrent modeling.

3. **Decoder**  
   Converts the masked encoded features back into enhanced waveform audio.

The original DPSNN is designed for approximately 5 ms latency under the low-latency configuration.

### Rhythm-DPSNN

Rhythm-DPSNN modifies selected spiking neurons using rhythm masks. Each rhythm neuron receives a periodic ON/OFF modulation signal controlled by:

- `cycle`: the period length of the rhythm.
- `duty_cycle`: the fraction of the cycle where the neuron is active.
- `phase`: the temporal offset of the rhythm.

A simplified view of the modulation is:

```text
ON state:
    normal neuron update and spike generation

OFF state:
    spike suppressed
    membrane update optionally frozen
```

The rhythm mechanism is tested in two main places:

1. **SRNN rhythmization**  
   Replaces ALIF neurons in the SRNN separator block with RhythmALIFNode.

2. **SCNN rhythmization**  
   Replaces PLIF neurons in the SCNN separator block with RhythmPLIFNode.

## Repository Structure

```text
.
├── audio_demos/
│   ├── vctk/
│   └── audio_demos_structure.txt
│
├── dpsnn/
│   ├── data/
│   │   ├── augment.py
│   │   ├── data_utils.py
│   │   ├── dnsmos.py
│   │   ├── hdf5_prepare.py
│   │   ├── metrics.py
│   │   ├── sig_bak_ovr.onnx
│   │   ├── voicebank_prepare.py
│   │   └── wave_dataset2.py
│   │
│   ├── layers/
│   │   ├── accelerating.py
│   │   ├── sdr.py
│   │   ├── sequential.py
│   │   ├── spike_activations.py
│   │   ├── spike_neuron.py
│   │   ├── spike_neurons.py
│   │   ├── srnn.py
│   │   └── surrogate.py
│   │
│   └── models/
│       └── dp_binary_net.py
│
├── egs/
│   └── voicebank/
│       ├── epoch=478-val_loss=81.5449-val_sisnr=-18.4556.ckpt
│       ├── vctk_trainer.py
│       └── vctk.yaml
│
├── figures/
├── installation.txt
└── README.md
```

## Important Branch Note

This project uses Git branches to separate experimental variants rather than keeping all variants active in one codebase.

Each experimental branch modifies the relevant model or neuron implementation for one specific experiment. The main files affected across branches are usually:

- `dpsnn/models/dp_binary_net.py`
- `dpsnn/layers/spike_neurons.py`

The file `dpsnn/layers/spike_neuron.py` was not modified for the rhythm experiments.

Before running an experiment, make sure you are on the correct branch:

```bash
git branch
git checkout <branch-name>
```

The active architecture is determined by the code in the currently checked-out branch, especially the model and neuron definitions imported by `egs/voicebank/vctk_trainer.py`.

## Experimental Branches

Different branches correspond to different DPSNN or Rhythm-DPSNN configurations, including:

- Plain DPSNN baseline.
- Rhythm-ALIF in the SRNN separator block.
- Rhythm-PLIF in the SCNN separator block.
- Deterministic rhythm assignment.
- Random rhythm assignment.
- Frozen adaptation state during OFF rhythm steps.
- Non-frozen adaptation state during OFF rhythm steps.
- Higher-duty-cycle rhythm settings.
- Learnable rhythm settings.

This branch-based organization makes it easier to compare experiments without mixing several incompatible implementations in one active codebase.

## Installation

Follow the setup in `installation.txt`.

```bash
conda create --name dpsnn python=3.11.5
conda activate dpsnn

conda install pytorch-cuda==11.8 pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 -c pytorch -c nvidia
pip install -r requirements.txt

pip install --editable .
```

## Dataset

The main dataset used in this project is VoiceBank+VCTK.

The configuration file is located at:

```text
egs/voicebank/vctk.yaml
```

The default dataset path in the config is:

```yaml
data_folder: ./voicebank
```

The expected generated files are:

```yaml
csv_train: ${save_folder}/train.csv
csv_valid: ${save_folder}/valid.csv
csv_test: ${save_folder}/test.csv

hdf5_train: ${save_folder}/train.hdf5
hdf5_valid: ${save_folder}/valid.hdf5
hdf5_test: ${save_folder}/test.hdf5
```

The default output and save folders are:

```yaml
output_folder: ${data_folder}/results
save_folder: ${output_folder}/save
```

Update the dataset paths inside `egs/voicebank/vctk.yaml` before training.

## Training

Move into the VoiceBank experiment folder:

```bash
cd egs/voicebank
```

### Train Plain DPSNN Baseline

```bash
python -u vctk_trainer.py \
  --config vctk.yaml \
  -L 80 \
  --stride 40 \
  -N 256 \
  -B 256 \
  -H 256 \
  --context_dur 0.01 \
  --max_epochs 500 \
  -X 1 \
  --lr 1e-2
```

### Train Rhythm-DPSNN

Switch to the correct rhythm branch first:

```bash
git checkout <rhythm-branch-name>
cd egs/voicebank
```

Then run the training command:

```bash
python -u vctk_trainer.py \
  --config vctk.yaml \
  -L 80 \
  --stride 40 \
  -N 256 \
  -B 256 \
  -H 256 \
  --context_dur 0.01 \
  --max_epochs 500 \
  -X 1 \
  --lr 5e-3
```

The exact rhythm behavior depends on the checked-out branch. For example, one branch may rhythmize the SRNN ALIF neurons while another may rhythmize the SCNN PLIF neurons.

## Default Configuration

The default VoiceBank config includes:

```yaml
sample_rate: 16000
frame_dur: 1.0
context_dur: 0.01
delay_dur: 0.00
max_frames: 5
batch_size: 1024
num_workers: 1
```

The trainer section includes:

```yaml
trainer:
  max_epochs: 1
  num_nodes: 1
  accelerator: gpu
  limit_train_batches: 0.01
  devices: 1
```

The optimizer section includes:

```yaml
optim:
  name: adam
  lr: 1e-2
  T_max: 64
```

For full training, change the trainer settings as needed. The uploaded config uses a limited training setup, which is useful for debugging but not for final model quality.

## Testing / Inference

To evaluate a trained checkpoint:

```bash
python -u vctk_trainer.py \
  --config vctk.yaml \
  -L 80 \
  --stride 40 \
  -N 256 \
  -B 256 \
  -H 256 \
  --context_dur 0.01 \
  --max_epochs 500 \
  -X 1 \
  --lr 1e-2 \
  --test_ckpt_path ./epoch=478-val_loss=81.5449-val_sisnr=-18.4556.ckpt
```

The checkpoint naming format is controlled by:

```yaml
checkpoint:
  monitor: 'val_loss'
  save_top_k: 3
  save_last: True
  filename: '{epoch}-{val_loss:.4f}-{val_sisnr:.4f}'
```

## Evaluation Metrics

This project evaluates both speech enhancement quality and computational efficiency.

### Speech Quality Metrics

- SI-SNR
- PESQ
- STOI
- DNSMOS OVRL
- DNSMOS SIG
- DNSMOS BAK

### Efficiency Metrics

- Total FLOPs
- Effective SynOPS
- SynOPS-delay product
- Estimated energy in Joules
- Spike power proxy
- Spike PDP proxy
- Per-module event firing rates

The efficiency metrics are proxy measurements. They are mainly useful for relative comparison between model variants. They should not be interpreted as exact hardware energy measurements unless the model is deployed and measured on actual neuromorphic hardware.

## Main Experimental Variants

### 1. Plain DPSNN Baseline

The baseline model keeps the original DPSNN architecture:

- Encoder: convolutional time-domain encoder.
- Separator: SCNN + SRNN.
- SCNN neuron: PLIF.
- SRNN neuron: ALIF.
- Decoder: convolutional decoder.

This is the main reference point for all rhythm experiments.

### 2. Rhythm-ALIF in SRNN

This variant replaces the ALIF neuron in the SRNN separator block with RhythmALIFNode.

Unchanged components:

- Encoder.
- Decoder.
- SCNN.
- Readout layer.

Rhythm behavior:

```text
ON state:
    normal ALIF update

OFF state:
    membrane frozen
    spike forced to 0
```

Several rhythm assignment strategies were tested:

- Deterministic linear spacing.
- Random uniform sampling.
- Frozen adaptation state during OFF steps.
- Non-frozen adaptation state during OFF steps.
- Milder rhythm ranges.
- Higher duty-cycle variants.
- Learnable rhythm variants.

The most stable fixed-rhythm variant used random rhythm assignment in the SRNN block.

### 3. Rhythm-PLIF in SCNN

This variant replaces the PLIF neuron in the SCNN separator block with RhythmPLIFNode.

Unchanged components:

- Encoder.
- Decoder.
- SRNN with standard ALIF dynamics.
- Readout layer.

This experiment showed that rhythm placement matters. Rhythm gating in the SCNN block slightly reduced the power proxy in one run but degraded speech enhancement quality. This suggests that rhythm gating can disrupt early temporal feature extraction when applied too early in the separator.

## Example Results

The following table summarizes the main comparison runs. Values may differ slightly depending on checkpoint selection, PyTorch version and training settings.

| Model | SI-SNR | PESQ | STOI | DNSMOS OVRL | DNSMOS SIG | DNSMOS BAK | Effective SynOPS | Spike Power Proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Noisy input | - | 1.971 | 0.921 | 2.684 | 3.324 | 3.111 | - | - |
| Plain DPSNN | 17.68 | 2.186 | 0.921 | 2.682 | 3.154 | 3.447 | 59.99M | 16.37M |
| Rhythm-DPSNN SRNN | 17.49 | 2.027 | 0.921 | 2.755 | 3.186 | 3.572 | 60.60M | 16.97M |
| Rhythm-DPSNN SCNN | 16.46 | 1.862 | 0.919 | 2.516 | 3.294 | 2.817 | 58.24M | 14.62M |

## Current Findings

The main findings from the experiments are:

1. Rhythm modulation can be integrated into DPSNN without breaking the training and evaluation pipeline.
2. Rhythm placement matters strongly.
3. Rhythm-ALIF in the SRNN block is more stable than Rhythm-PLIF in the SCNN block.
4. Rhythm-PLIF in the SCNN block can slightly reduce the power proxy but harms speech enhancement quality.
5. Random rhythm assignment is more stable than deterministic linear rhythm assignment.
6. Freezing the ALIF adaptation state during OFF steps harms performance.
7. Higher firing activity does not automatically improve speech quality.
8. Fixed SRNN rhythm improves DNSMOS background quality in some runs but does not improve PESQ or SI-SNR over the baseline.
9. Efficiency gains were not consistently achieved in the current SRNN rhythm configuration.

## Audio Demos

Audio examples are stored in:

```text
audio_demos/
```

The audio demos are kept in the same format as the original DPSNN project.

Suggested organization:

```text
audio_demos/
├── vctk/
│   ├── noisy/
│   ├── clean/
│   ├── baseline_dpsnn/
│   └── rhythm_dpsnn/
└── audio_demos_structure.txt
```

## Reproducibility Notes

For fair comparison across branches:

1. Use the same dataset split.
2. Use the same sample rate.
3. Use the same checkpoint selection rule.
4. Use the same evaluation scripts.
5. Use the same low-latency configuration.
6. Compare runs with matched training budgets.
7. Record the exact branch name and commit hash for every experiment.

Recommended logging:

```bash
git branch --show-current
git rev-parse HEAD
```

## Citation

If you use the original DPSNN code or model in an academic context, cite:

```bibtex
@article{sun2024dpsnn,
  title={DPSNN: spiking neural network for low-latency streaming speech enhancement},
  author={Sun, Tao and Boht{\'e}, Sander},
  journal={Neuromorphic Computing and Engineering},
  volume={4},
  number={4},
  pages={044008},
  year={2024},
  publisher={IOP Publishing}
}
```

If you use the rhythm modulation extension, also cite the neural oscillation / Rhythm-SNN work that inspired the modification:

```bibtex
@article{yan2025efficient,
  title={Efficient and robust temporal processing with neural oscillations modulated spiking neural networks},
  author={Yan, Yinsong and Yang, Qu and Wu, Yujie and Liu, Hanwen and Zhang, Malu and Li, Haizhou and Tan, Kay Chen and Wu, Jibin},
  journal={Nature Communications},
  volume={16},
  pages={8651},
  year={2025}
}
```

## License

Add the license used by the original DPSNN repository and any additional license requirements for this modified project.

## Acknowledgements

This project builds on the original DPSNN implementation by Tao Sun and Sander Bohté. The rhythm modulation experiments are inspired by neural oscillation based SNN research and were developed as part of a thesis project on neural oscillations in spiking neural networks for low-latency speech enhancement.
