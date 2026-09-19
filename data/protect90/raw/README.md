# PROTECT-90: A Fault Dataset for Power System Protection

**Alternative title:** 90 kV Double-Line EMT Fault Waveform Dataset for High-Voltage Protection  
**Subtitle:** Open, Standardized Voltage and Current Waveforms for Reproducible Protection Research  
**Version:** 1.0.0  
**License:** CC BY 4.0  

---

## Overview

**PROTECT-90** is an open high-voltage fault waveform dataset for reproducible research in **power system protection**, **transient waveform analysis**, and **learning-based fault analysis**.

The dataset contains **9,022 electromagnetic transient (EMT) simulation episodes** generated on a standardized **90 kV double-line transmission topology**. Each episode provides **1 second of synchronized three-phase voltage and current waveforms** sampled at **6.4 kHz**, corresponding to **128 samples per 50 Hz cycle**.

Measurements are recorded at **eight protection-relevant locations**, resulting in **48 synchronized waveform channels** per episode. All episodes are released as **raw instantaneous waveforms** without feature engineering and are accompanied by structured, machine-readable metadata.

PROTECT-90 was developed within the DFG-funded project **Coordinated Power System Protection using Machine Learning (Netzschutz-KI)**, project number **535389056**, at Friedrich-Alexander-Universität Erlangen-Nürnberg.

---

## Start Here

The Zenodo record provides the dataset files and persistent DOI.

Additional resources:

- **Companion repository, documentation, citation guidance, and guided notebook:**  
  <https://github.com/julianoelhaf/protect90-dataset>

- **Paper preprint:**  
  <https://arxiv.org/abs/2606.24298>

- **Project context:**  
  <https://lme.tf.fau.de/research/research-groups/data-processing-for-utility-infrastructure/ai-grid-protection/>

The companion repository is the recommended entry point for new users. It links the dataset, paper, citation information, and a guided notebook for inspecting the waveform files and metadata.

---

## Dataset Summary

- **Episodes:** 9,022 EMT simulation episodes
- **Topology:** 90 kV double-line transmission system
- **Duration:** 1.0 s per episode
- **Sampling rate:** 6.4 kHz
- **Samples per episode:** 6,400 time steps
- **Samples per 50 Hz cycle:** 128
- **Measurement locations:** 8
- **Waveform channels:** 48
- **Signals:** three-phase voltages and currents
- **Signal type:** raw instantaneous secondary voltage and current waveforms
- **Feature engineering:** none
- **Metadata:** structured CSV file
- **Missing values:** none in the released waveform files

Each episode represents one physically consistent simulated short-circuit scenario.

---

## File Structure

The Zenodo record contains:

```text
hv_double_line_90kv_labels.csv
hv_double_line_90kv_preprocessed_data.zip
README.md
```

After extracting the waveform archive:

```text
preprocessed_data/
  {sample_id}_sample_hv_double_line_90kv.pkl
```

Each `.pkl` file contains one EMT episode as a pandas DataFrame with shape:

```text
(6400, 49)
```

The 49 columns consist of:

- 48 waveform channels
- 1 time column: `time_s`

---

## Measurement Structure

For each episode, three-phase voltage and current waveforms are recorded at eight protection-relevant measurement locations.

Each measurement location contributes six channels:

```text
V_A, V_B, V_C, I_A, I_B, I_C
```

Across eight locations, this results in:

```text
8 locations × 6 channels = 48 waveform channels
```

All signals are synchronized and sampled at 6.4 kHz.

The waveform values correspond to secondary-side voltage and current quantities, consistent with digital fault recording practice.

---

## Metadata

The file

```text
hv_double_line_90kv_labels.csv
```

contains structured metadata for all episodes.

The metadata includes, among others:

- sample identifier
- fault type
- faulted line segment
- normalized fault location
- fault resistance
- fault inception time
- randomized line parameters
- load operating points
- external grid parameters
- voltage magnitude and phase angle
- topology switching states

Formatting:

- Encoding: UTF-8
- Separator: comma
- Decimal separator: period

---

## Domain Randomization

PROTECT-90 was generated under physically constrained domain randomization.

The randomized quantities include:

- line parameters
- load operating points
- external grid short-circuit strength
- voltage magnitude
- voltage phase angle
- parallel line switching states
- fault type
- fault resistance
- spatial fault location
- fault inception time

All simulated scenarios are subject to physical and numerical plausibility checks, including load-flow convergence, EMT numerical stability, and plausible parameter ranges.

This design supports robustness and domain-shift studies while preserving a controlled and reproducible benchmark setting.

---

## Intended Use

PROTECT-90 can be used for research on:

- fault detection
- fault classification
- fault localization
- faulted-line identification
- coordinated protection
- transient waveform analysis
- reduced-observability studies
- robustness and domain-shift evaluation
- reproducible benchmarking of machine-learning and signal-processing methods

The dataset is intended to support transparent, standardized, and cross-study comparable evaluation.

---

## Recommended Evaluation Practice

No predefined train/test split is provided.

Users should define task-specific splits explicitly and report them clearly. For learning-based experiments, episode-wise partitioning is recommended to avoid leakage between windows or derived samples from the same EMT episode.

When reporting results, users should describe at least:

- task definition
- input observability, for example single location, line terminal pair, or all locations
- waveform window length, if windowing is used
- preprocessing and normalization
- train/validation/test split strategy
- evaluation metrics
- whether results are episode-based or window-based

---

## Scope and Limitations

PROTECT-90 is a simulated EMT dataset.

It is intended as a transparent and reproducible research benchmark. It is not a replacement for utility-specific protection studies, field validation, hardware-in-the-loop testing, or certified protection functions.

The dataset should be interpreted in the context of its documented topology, simulation assumptions, parameter ranges, and measurement configuration.

---

## Citation

If you use PROTECT-90, please cite the dataset DOI and, where appropriate, the accompanying paper.

Dataset DOI:

```text
10.5281/zenodo.18418330
```

Paper preprint:

```text
https://arxiv.org/abs/2606.24298
```

For up-to-date BibTeX entries, please see the companion repository:

```text
https://github.com/julianoelhaf/protect90-dataset
```

---

## Versioning

```text
Current version: 1.0.0
```

This is the first stable public release of PROTECT-90. Semantic versioning is used. Please cite the exact dataset version used in your experiments.

---

## License

PROTECT-90 is released under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

Users may share and adapt the dataset, provided appropriate credit is given.

---

## Contact

For technical questions about using the dataset, please open an issue in the companion repository:

<https://github.com/julianoelhaf/protect90-dataset>

For academic, citation, or collaboration-related questions, please contact:

- Georg Kordowich — georg.kordowich [at] fau [dot] de
- Julian Oelhaf — julian.oelhaf [at] fau [dot] de

Institutions:

- Friedrich-Alexander-Universität Erlangen-Nürnberg
  [https://www.fau.de/](https://www.fau.de/)

- Pattern Recognition Lab, FAU
  [https://lme.tf.fau.de/](https://lme.tf.fau.de/)

- Institute of Electrical Energy Systems, FAU
  [https://www.ees.tf.fau.de/](https://www.ees.tf.fau.de/)

- Ostbayerische Technische Hochschule Amberg-Weiden
  [https://www.oth-aw.de/](https://www.oth-aw.de/)
