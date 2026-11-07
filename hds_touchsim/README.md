# hds_touchsim

HDS pattern (binary 5×3 tensor) → TouchSim → spike counts → similarity between digits.

```
hds_touchsim/
  hds.py        HDSSpec, digit patterns, pin geometry, fit_to_region      (no TouchSim needed)
  adapter.py    hand rescaling, pixel<->hand mm, stimulus, population     (takes the touchsim module as `ts`)
  evaluate.py   spike counts, Pearson similarity
  viz.py        one plot: hand outline + HDS
scripts/run_pipeline.py   full run for digits 0-9, saves out/results.npz
tests/                    pytest (adapter tests use tests/fake_touchsim.py, no TouchSim needed)
hds_touchsim_clean.ipynb  the same code in one notebook
```

```
pip install -r requirements.txt
pytest
python scripts/run_pipeline.py --density 1.0
```

The hand is rescaled to 189 mm (`REAL_HAND_LENGTH_MM`); the HDS keeps its real size (50×80 mm, pin Ø10 mm,
pitch 15 mm, depth 1 mm). Pin positions and radius are divided by `hand_scale` when sent to TouchSim.
Afferents are subsampled to SA1 : RA : PC = 2 : 4 : 1 (`make_population(class_ratio=None)` to disable).
