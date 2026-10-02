# Walking in Every Direction

A Robinson/Star-tiling scaffold for sampling the walking directions of
simulated pedestrians, headless data generation with a Unity simulation, and
PyTorch training of a footstep-detection network on the resulting synthetic
data, which detects real footsteps on SensFloor sensor mats of the same shape.

Code and data for the experiments in

> J. Dünnweber, J. Geyer: *Walking in Every Direction: Aperiodic Data Enrichment
> and Anisotropic Kernels for Floor-Sensor Gait Analysis.* Submitted to the
> Journal of Ambient Intelligence and Smart Environments (JAISE).

It extends our earlier work on footstep detection with floor sensor mats
(J. Geyer, M. Melzer, J. Dünnweber, A. Sarkar: *Human Step Movement Imitation
for Training Privacy-Aware Intelligent Environments*, IEEE IE 2026,
doi:10.1109/IE69249.2026.11539056).

## What this code is good for

Floor sensor mats (here: SensFloor) detect footsteps without cameras. An
instance segmentation network assigns simultaneously active sensor
triangles to individual feet; it is trained mainly on synthetic footsteps
from a Unity simulation of walking humans driven by motion capture data.
This repository lets you

- **train and evaluate the network** on the four conditions of Table 3 in
  the paper: scarce real data only, synthetic data with the earlier
  narrow-range heading sampler (±60°), with a naive uniform 360° sampler, and
  with the direction-unbiased Robinson/Star-tiling sampler combined with
  direction-sensitive (anisotropic) convolution kernels;
- **repeat the training with several random seeds**, including an extra
  condition (Robinson headings with isotropic kernels) that separates the
  effect of the heading sampler from that of the kernels;
- **regenerate the synthetic training data** with the Unity simulation
  (optional, all data used in the paper is included);
- **verify** the property of the Robinson tiling the heading sampler relies on.

## Repository layout

```
code/                     PyTorch code
  train.py, evaluate.py   entry points (training / precision-recall at IoU 0.50, 0.75, 0.95)
  archs/                  isotropic network and the anisotropic variant
                          (oriented_conv2d.py: bank of five oriented 1x3 kernels)
  losses/                 discriminative loss (pixel embeddings)
  postprocess/            mean-shift clustering and IoU matching
  split_real_data.py      fixed split of the 49 real recordings (33 train / 16 test)
data/                     all data used in the paper (see below)
data_generation/
  generate_headings.py    heading samplers (naive 360°, Robinson)
  verify_robinson_axes.py numerical check of the Robinson heading sampler
  installUnity.sh         installs Unity Hub on Fedora/RHEL
  unity/                  headless data generation with the Unity simulation
run_smoketest.sh          2-epoch check that the environment works
run_table3_final.sh       trains and evaluates the four Table 3 conditions
run_table3_seeds.sh       the same with seeds 1-3, plus Robinson + isotropic
```

## Data

Every line of a `data/*.txt` file is one training example:
`<800-character activation string>:<mask>,<mask>,...,` where the
activation string is the 20 x 40 grid of sensor triangles (row-major, 0/1)
and each mask is another 800-character 0/1 string marking the triangles of
one foot.

| File | Content |
|---|---|
| `real_scarce.txt` | 49 hand-annotated real recordings (bus and grocery store) |
| `real_scarce_train.txt`, `real_heldout_test.txt` | fixed split of the above: 33 for training (condition 1), 16 held out as the common test set of all conditions |
| `train_synthetic_old_range.txt` | 25918 synthetic examples, narrow-range heading sampler (condition 2) |
| `train_naive_360.txt` | 25918 synthetic examples, uniform 360° headings (condition 3) |
| `train_robinson.txt` | 25918 synthetic examples, Robinson/Star headings (condition 4) |
| `val_synthetic.txt` | synthetic validation set used during training |
| `smoketest/` | four examples for `run_smoketest.sh` |

## How to run it

Requirements: Python 3.11 or newer, PyTorch, scikit-learn, NumPy, Matplotlib.
A CUDA GPU is strongly recommended; install the PyTorch build that matches
your CUDA version (<https://pytorch.org/get-started/locally/>).

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./run_smoketest.sh                          # seconds; checks the environment

EPOCHS=200 DEVICE=cuda ./run_table3_final.sh
cat results/table3_rows.tex                 # Table 3 rows (precision/recall at IoU 0.5)
cat results/*/eval.csv                      # all IoU thresholds
```

With two GPUs, run disjoint rows in parallel, e.g.
`CUDA_VISIBLE_DEVICES=0 ROWS="1 3" ./run_table3_final.sh` and
`CUDA_VISIBLE_DEVICES=1 ROWS="2 4" ./run_table3_final.sh`.
Finished rows are skipped when a script is restarted (`FORCE=1` retrains).

Repeated runs with seeds (results and mean ± standard deviation in
`results_seeds/summary.txt`):

```bash
./run_table3_seeds.sh        # env: CONDS, SEEDS, GPUS, PER_GPU, EPOCHS
```

For reference, on an AMD Ryzen Threadripper PRO 7965WX with two NVIDIA RTX
4000 Ada GPUs, 200 epochs took about 52-58 minutes per synthetic condition
(one run per GPU) and under two minutes for condition 1. Single training
runs on the 16 held-out recordings vary noticeably with the random seed;
compare conditions with `run_table3_seeds.sh` rather than single runs.

## Regenerating the synthetic data (optional)

The data for conditions 3 and 4 was generated with the SensFloorSimulation
Unity project (Unity 2022.3.26f1) of the IE 2026 paper, which is not part of
this repository; please contact the authors if you need it.
`data_generation/unity/generate_table3_data.sh` copies that project, patches
the copy so that each simulated pedestrian's heading is read from a
pre-generated file (`apply_unity_patch.py`), runs the simulation headless
until 25918 examples with at least one foot exist, and writes
`data/train_naive_360.txt` / `data/train_robinson.txt`:

```bash
sudo data_generation/installUnity.sh        # Fedora/RHEL; then install the editor via Unity Hub
PROJECT=/path/to/SensFloorSimulation data_generation/unity/generate_table3_data.sh both
```

The simulation runs in real time (about three hours per condition); a
watchdog restarts Unity if it stops producing output, and interrupted runs
resume.

`data_generation/unity/Table3DataGen.cs` is the only C# file in the
repository: a small Unity editor script that `apply_unity_patch.py` copies
into the project's `Assets/Editor/` folder. Unity can only run C# code, so
this script is the entry point that `generate_table3_data.sh` calls in batch
mode (`-executeMethod Table3DataGen.Run`) to open the simulation scene and
start it. The changes to the simulation itself are applied as text patches
by `apply_unity_patch.py`. None of this is needed for training and
evaluation with the data included in this repository.

`python3 data_generation/verify_robinson_axes.py` confirms that the axes of
symmetry of all triangles in a wheel-seeded Robinson tiling point in the ten
directions k·36° with equal frequency, which makes the Robinson heading
sampler equivalent to sampling a triangle of the tiling uniformly. The
tiling is built with the standard Robinson-triangle subdivision as presented
in J. Preshing, [*Penrose Tiling Explained*](https://preshing.com/20110831/penrose-tiling-explained/) (2011).

## Authorship and use of AI assistance

The network architecture, loss function, data loading, training loop,
post-processing, and the Unity simulation of walking humans were developed by
Johannes Geyer and Jan Dünnweber. The direction-sensitive kernel bank, the
experiment scripts, the headless data generation for the Robinson and uniform
360° conditions, and the verification of the Robinson heading sampler were
developed with the help of the AI coding assistant Claude Code (Anthropic),
under the direction of the authors, who designed the experiments,
reviewed the code, and checked all results reported in the paper.

## License

MIT, see `LICENSE`.
