# DFODE-kit

DFODE-kit is a Python toolkit for accelerating combustion chemistry integration with deep learning. It supports the end-to-end workflow of:

1. sampling thermochemical states from canonical flame cases,
2. augmenting and labeling datasets,
3. training neural network surrogates for stiff chemistry integration,
4. validating and deploying them in DeepFlame-based CFD workflows.

## What this docs site covers

- [Research method catalogue](research-method-catalogue.md): canonical names,
  diagrams and mappings to saved experiment IDs.

- **Getting Started**: environment setup and installation
- **CLI**: current `dfode-kit` commands and their purpose
- **Canonical Case Initialization**: preset-based case setup with preview/apply/config workflows
- **Runtime Configuration and Case Execution**: persistent machine-local environment config plus reproducible case launching
- **Data Augmentation**: the minimal preset-driven augment CLI, including preview/apply/config round-trip
- **Data Preparation and Training Workflow**: the current artifact flow from sampled HDF5 to labeled datasets and models
- **Architecture**: repo layout and current refactor direction
- **Tutorials and Workflow**: how to think about the DFODE pipeline
- **Agent Docs**: operational guidance for coding agents and maintainers

## Project goals

The current development direction is to make DFODE-kit:

- easier for humans and coding agents to use,
- more reproducible for experiment workflows,
- more modular for new model architectures and training algorithms,
- more robust through contracts, tests, and CI.

## Repository

- GitHub: [deepflame-ai/DFODE-kit](https://github.com/deepflame-ai/DFODE-kit)
- DeepFlame docs: [deepflame.deepmodeling.com](https://deepflame.deepmodeling.com/en/latest/)

<!-- docs-pages trigger -->
