# Test-time physics-informed fine-tuning

Adapting a neural PDE surrogate to a shifted regime using only unlabeled initial
conditions and the residual of an equation whose coefficient is wrong.

<p align="center"><img src="assets/approach_overview.png" width="760"></p>

## Findings

A 3D U-Net surrogate pretrained on Kolmogorov flow at Re100 is deployed at Re500 and
fine-tuned on five unlabeled initial conditions, with the source viscosity left in the
residual.

- **A misspecified residual still helps.** Adaptation lengthens the prediction horizon
  at every resolved scale and recovers 43% of the gap to a model trained on labeled
  Re500 data.
- **The correct viscosity adds nothing.** Putting the true target viscosity into the
  residual gives no further gain.
- **The gain needs a budget.** It appears only after more than 200 steps. Below that,
  the largest scales regress first, and the residual gives no warning.

<p align="center">
  <img src="assets/budget.png" width="49%">
  <img src="assets/spectrum.png" width="49%">
</p>

## Run

    pip install -r requirements.txt
    python -m msc.tta.adapt.adapt experiment=unet_nu100

## Workflow

- `main`: the only branch experiments run from. The server checkout stays clean on `main`
  and is updated with `git pull`; files are never copied to it by hand.
- `dev`: integration branch. Every change goes through a feature branch and a PR into
  `dev`; tests run on the server before merge.
- Once a change is settled on `dev`, a PR `dev` → `main` releases it, then the server pulls.
- Commits stage code and configs (`.py`, `.yaml`); research notes (`.md`) stay local unless
  they are repo documentation such as this file.
- Authorship: commits and PRs carry the human author only. No AI co-author trailers and no
  "generated with" lines, in commit messages, PR titles, PR bodies or comments.
