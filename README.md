# Activation-gradient metamodel

This repository holds the code behind the sprint talk on a gradient metamodel. The metamodel is a Qwen3-14B reader that takes the activation-space gradient of a training loss as input (one vector per layer, injected into its residual stream) and answers natural-language questions about the training data that produced that gradient. It is trained with SFT on labelled document bundles and then with RL against an LLM judge. We then apply it to held-out behaviour bundles and to the per-step updates of a fine-tuned student model (subliminal learning).

Data, gradient stores, model weights and checkpoints are not included. The code expects them under a root directory given by the `IIF_BASE` environment variable.

## Pipeline

1. Gradient extraction. `scripts/beh_reader/extract.sh` runs `main_run/extract.py` (with `main_run/agm/`) over the behaviour bundles and writes a sharded gradient and activation store. `main_run/merge_chunks.py` merges chunked stores. The C1 store of the pretraining-mix documents is joined and split by `main_run/c1/` and `main_run/grad_io.py`.
2. Reader directory. `main_run/beh_reader/build_reader_dir.py` and `build_rl_dir.py` build the SFT and RL directories (C1 bundles plus behaviour bundles). `check_reader_dir.py` and `check_rl_dir.py` verify them. `write_configs.py` writes the training configs in `configs/beh_reader/`. The EM reader counterparts are in `main_run/em_reader/`.
3. SFT. `main_run/reader/train.py` with a `configs/beh_reader/scratch_*.yaml` or `configs/reader/scratch_*.yaml` config. `main_run/reader/select_checkpoint.py`, `nll_eval.py` and `judge_curve.py` pick the checkpoint.
4. RL. `main_run/reader/rl_train.py` (with `rl_data.py`, `rl_eval.py`, `rl_judge.py`) and a `configs/*/rl_*.yaml` config. `scripts/beh_reader/chain.sh` runs steps 2 to 5 end to end for the behaviour reader, and `scripts/beh_reader/run_L3.sh` and `activation_twin.sh` do the same for the 3-layer gradient and activation metamodels.
5. Reading and judging. `main_run/reader/e2_generate.py` samples answers and the first-token P(Yes). `main_run/beh_reader/export_inputs.py`, `read_eval.py`, `answer_tags.py`, `level_tags.py` and `level_summary.py` produce and grade the open answers. Judges are in `beh_judges.py` and `main_run/em_reader/em_judges.py`.
6. Student experiments. `scripts/newset/train.sh` trains the subliminal-learning students (`main_run/train_subliminal.py`, `online_save.py` stores per-step gradients). `main_run/adamfx.py` builds the per-step Adam update terms. `scripts/cleanbatch/` and `main_run/train_cleanbatch.py` compare real and clean batches. `scripts/blockmix/filter.sh` with `main_run/train_blockmix_filter.py` filters training batches by the metamodel score. `scripts/beh_reader/harm_generalisation.sh` and `already_learned.sh` train the students for slides 18 and 19.

## Slides and scripts

| Slide | Title | Data scripts | Figure script |
|---|---|---|---|
| 6 | Accumulated gradients reveal animal-specific SAE features | `main_run/sae_grad_enrichment.py` | `main_run/plot_sae_grad_range_for_slide.py`, `plot_sae_grad_window_counts_for_slide.py` |
| 7 | Fitted directions reveal animal-specific SAE features | `main_run/fit_vector.py`, `main_run/sae_feature_enrichment.py` | `main_run/plot_sae_feature_enrichment.py` |
| 8 | Fitted residual biases causally shift animal preference | `main_run/fit_vector.py` | `main_run/plot_fitvec_steering_slide.py` |
| 10 | The input is the activation-space gradient of the training loss | `scripts/beh_reader/extract.sh`, `main_run/extract.py` | none |
| 11 | Labels are answers that an LLM labeler writes | `main_run/beh_reader/build_reader_dir.py`, `main_run/c1/materialize.py` | none |
| 13 | An open answer is graded on four cumulative levels | `main_run/beh_reader/level_tags.py`, `answer_tags.py` | none |
| 14 | How well does the gradient metamodel read held-out bundles? | `scripts/beh_reader/chain.sh`, `main_run/beh_reader/level_tags.py`, `level_summary.py`, `level_shuffled.py` | `main_run/plot_level_tags.py` |
| 15 | What the metamodel says, at four, two and zero levels passed | `main_run/beh_reader/level_tags.py`, `level_summary.py` | none |
| 16 | Which metamodel recovers the training data better? | `scripts/beh_reader/run_L3.sh`, `activation_twin.sh`, `ao_heldout.sh` | `main_run/plot_level_tags.py` |
| 17 | Does reading more layers help? | same as slide 16 | `main_run/plot_level_tags.py` |
| 18 | Does it predict change beyond the planted domain? | `scripts/beh_reader/harm_generalisation.sh`, `broad_change_questions.sh` | `main_run/plot_broad_change_slide_split.py` |
| 19 | Does it stop flagging a behaviour the model has learned? | `scripts/beh_reader/already_learned.sh`, `new_readers_suite.sh` | `main_run/plot_already_learned_bars_per_behaviour.py` |
| 20 | Can the activation metamodel read gradients at all? | `main_run/beh_reader/cross_input.py` | `main_run/plot_level_tags.py` |
| 22 | Reading subliminal learning one AdamW step at a time | `main_run/adamfx.py`, `main_run/cleanbatch_probe.py` | `main_run/plot_direct_write_block.py` |
| 23 | Which part of the Adam update carries the signal? | `scripts/newset/train.sh`, `scripts/cleanbatch/chain.sh`, `variants_chain.sh` | `main_run/plot_cleanbatch_bars.py` |
| 24 | Does the update read as dolphin before the student shows it? | `scripts/newset/train.sh`, `scripts/cleanbatch/chain.sh`, `extend.sh` | `main_run/plot_cleanbatch_full.py` |
| 25 | Does filtering by the metamodel's score stop the preference? | `scripts/blockmix/filter.sh`, `main_run/filter_onsets_one_rule.py` | `main_run/plot_filter_rank_probe.py` |
| 26 | Why does filtering only delay the preference? | `scripts/blockmix/filter.sh`, `main_run/plot_filter_rank_threshold.py` | `main_run/plot_filter_threshold_slide.py` |
| 28 to 33 | Gradient metamodel distinguishes training behaviours more strongly than the activation metamodel | `main_run/experiment2/` (`runtime.py`, `probe.py`, `analyse.py`), `configs/experiment2/` | `main_run/experiment2/plot_reader_delta_pyes_slide.py` |
| 28 to 33 | Document weighting nearly explains rare-behaviour detection | `main_run/ao_vs_reader/probe_compare.py`, `probe_tasks.py` | `main_run/plot_probe_weighting_slide.py` |
| 28 to 33 | Loss-weighted activations improve rare-behaviour detection | `scripts/experiment2/loss_weighted.sh`, `main_run/experiment2/loss_weighted.py` | `main_run/experiment2/plot_loss_weighted_delta_pyes_slide.py` |
| 28 to 33 | Output error nearly closes the logit-lens gap | `main_run/fwd_baselines/` (`extract_fwd.py`, `build_inputs.py`, `fwd_probe.py`, `fwd_lens.py`, `decide.py`), `main_run/ao_vs_reader/logit_lens.py` | `main_run/plot_forward_baseline_gap_slide.py` |
| 28 to 33 | Gradients distinguish teacher-generated data, but not uniquely by animal preference | `main_run/signal_act_probe.py` | `main_run/plot_act_probe_slide.py` |
| 28 to 33 | Loss-weighted activations fail to reveal the subliminal dolphin signal | `main_run/lossw_actdiff.py`, `lossw_cleanbatch.py`, `lossw_sae.py` | `main_run/plot_lossw_dolphin.py`, `plot_lossw_answers.py` |
| 28 to 33 | Knowing the target changes which method detects learning first | `main_run/blackbox_timing.py`, `scripts/candidates/score_subliminal.sh`, `rank_subliminal.sh` (`main_run/reader/candidate_score.py`, `main_run/candidate_rank.py`) | `main_run/plot_told_vs_not_told_slide.py` |
| 35 | The metamodel detects risky advice early, but misses emergent misalignment | `main_run/train_em.py`, `main_run/em/` (`eval.py`, `onset.py`, `make_ctx_data.py`, `beh_jobs.py`, `ctx_compare.py`, `judge_reader.py`), `scripts/em_beh/judge_all.sh` | `main_run/plot_em_beh_slide.py` |
| 36 | Gradients distinguish finished EM models, but the metamodel misses the distinction | `main_run/em/` (`dact_extract.py`, `dwx_inputs.py`, `ext_grad_inputs.py`, `pairwise_judge.py`, `pair_input_check.py`), `scripts/em_dact/judge_all.sh` | `main_run/plot_em_pair_slide.py` |
| 37 | The metamodel tracks test-related changes but fails to identify reward hacking | `main_run/rh/` (`make_runs.py`, `make_docs.py`, `half_inputs.py`, `judge_all.py`, `summary.py`), `main_run/ao_diff_read.py` | `main_run/rh/plot_slide.py`, `plot_slide_trajectory.py` |
| 38 | Many wrong answers repeat behaviours from the metamodel's own training set | `main_run/beh_reader/answer_tags.py`, `level_tags.py` | none |

The appendix slides on public-AO input tuning and on the text ceiling come from `main_run/beh_reader/rare_yesno.py` with `scripts/beh_reader/rare_yesno.sh` and `rare_yesno_ext.sh` (figure `main_run/plot_ao_prep_sweep_slide.py`), and from `main_run/experiment2/text_ceiling.py` and `loss_detector.py` (figure `main_run/experiment2/plot_text_ceiling_identification.py`). The behaviour-labelled bundles that slides 11 to 20 read are built by `main_run/behaviour/` (`registry.py`, `fetch.py`, `rows.py`, `qa.py`, `bundles.py`, `split.py`, `handoff.py`). The figure scripts of the slides above read the small summary files kept in `docs/figures/`, `docs/rh_results/` and `results/`; the run outputs they come from are not included.

Slides 23 to 26 use the older 40-layer metamodel trained on C1 and FineWeb RL only, with `configs/reader/scratch_L40.yaml` and `configs/reader/rl_scratch_L40_haiku_paper.yaml`.

## Environment

- `IIF_BASE` is the root for data, stores, runs and the Hugging Face cache (`$IIF_BASE/hf_home`). Scripts stop with an error when it is unset.
- API keys are read at run time from files under `$IIF_BASE/persistent/secrets/` (for example `anthropic_key` for the Haiku judges). No key is stored in this repository.
- The OpenRouter judges (`main_run/c1/score.py` and the scripts that reuse it) take `OPENROUTER_API_KEY` from the environment, a gitignored `.env`, or `$IIF_BASE/persistent/secrets/openrouter_key_2`.
- Python packages are listed in `requirements.txt`. The base model is `Qwen/Qwen3-14B`.
- Weights, checkpoints, gradient stores and datasets are not included. The C1 split manifest `main_run/c1/manifests/splits.json` (465 KB) is left out too, and `main_run/c1/build_manifest.py` writes it.

## Code versions

The SFT and reading code in `main_run/reader/` and `main_run/c1/` comes from the version used for SFT and reading runs. The RL files `main_run/reader/rl_*.py` and the RL configs come from the version used for RL runs. A few files existed in two differing versions, and each of those starts with a one-line comment that names the version kept.
