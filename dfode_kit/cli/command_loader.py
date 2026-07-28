import importlib
from collections import OrderedDict


_COMMAND_SPECS = {
    "generate-implicit-root-data": {
        "module": "dfode_kit.cli.commands.generate_implicit_root_data",
        "help": "Generate exact Backward-Euler root pairs by adaptive continuation.",
    },
    "evaluate-hybrid-integrators": {
        "module": "dfode_kit.cli.commands.evaluate_hybrid_integrators",
        "help": "Compare adaptive implicit solvers with conventional and neural guesses.",
    },
    "evaluate-implicit-warm-start": {
        "module": "dfode_kit.cli.commands.evaluate_implicit_warm_start",
        "help": "Evaluate neural warm starts for a Backward-Euler chemistry solve.",
    },
    'augment': {
        'module': 'dfode_kit.cli.commands.augment',
        'help': 'Perform data augmentation.',
    },
    'h52npy': {
        'module': 'dfode_kit.cli.commands.h52npy',
        'help': 'Convert HDF5 scalar fields to NumPy array.',
    },
    'generate-0d-sequences': {
        'module': 'dfode_kit.cli.commands.generate_0d_sequences',
        'help': 'Generate 0D constant-pressure reactor trajectory datasets.',
    },
    'generate-0d-suite': {
        'module': 'dfode_kit.cli.commands.generate_0d_suite',
        'help': 'Generate train/validation 0D reactor sequence suites.',
    },
    'generate-interval-pairs': {
        'module': 'dfode_kit.cli.commands.generate_interval_pairs',
        'help': 'Convert 0D sequence trajectories into variable-dt interval pairs.',
    },
    'generate-interval-thermo-features': {
        'module': 'dfode_kit.cli.commands.generate_interval_thermo_features',
        'help': 'Precompute chemical-potential affinity features for interval pairs.',
    },
    'init': {
        'module': 'dfode_kit.cli.commands.init',
        'help': 'Initialize canonical cases from explicit presets.',
    },
    'config': {
        'module': 'dfode_kit.cli.commands.config',
        'help': 'Manage persistent runtime configuration.',
    },
    'diagnose-0d-reactivity': {
        'module': 'dfode_kit.cli.commands.diagnose_0d_reactivity',
        'help': 'Diagnose reactive-state coverage in 0D sequence datasets.',
    },
    'design-hit-flame-case': {
        'module': 'dfode_kit.cli.commands.design_hit_flame_case',
        'help': 'Design a mechanism-aware 2D HIT premixed-flame validation case.',
    },
    'evaluate-latent-sequence': {
        'module': 'dfode_kit.cli.commands.evaluate_latent_sequence',
        'help': 'Evaluate a latent sequence model checkpoint.',
    },
    'evaluate-stoich-interval': {
        'module': 'dfode_kit.cli.commands.evaluate_stoich_interval',
        'help': 'Evaluate a variable-dt stoichiometric interval model.',
    },
    'export-hit-openfoam-u': {
        'module': 'dfode_kit.cli.commands.export_hit_openfoam_u',
        'help': 'Export a designed 2D HIT velocity field to an OpenFOAM U file.',
    },
    'benchmark-stoich-interval-runtime': {
        'module': 'dfode_kit.cli.commands.benchmark_stoich_interval_runtime',
        'help': 'Benchmark variable-dt interval checkpoint inference runtime.',
    },
    'label': {
        'module': 'dfode_kit.cli.commands.label',
        'help': 'Label data.',
    },
    'run-case': {
        'module': 'dfode_kit.cli.commands.run_case',
        'help': 'Run a DeepFlame/OpenFOAM case using stored configuration.',
    },
    'sample': {
        'module': 'dfode_kit.cli.commands.sample',
        'help': 'Perform sampling.',
    },
    'train': {
        'module': 'dfode_kit.cli.commands.train',
        'help': 'Train the model.',
    },
    'train-conserved-sequence': {
        'module': 'dfode_kit.cli.commands.train_conserved_sequence',
        'help': 'Train a hard atom-conserving sequence baseline.',
    },
    'train-stoich-sequence': {
        'module': 'dfode_kit.cli.commands.train_stoich_sequence',
        'help': 'Train a stoichiometric reaction-flux sequence baseline.',
    },
    'train-stoich-interval': {
        'module': 'dfode_kit.cli.commands.train_stoich_interval',
        'help': 'Train a variable-dt stoichiometric interval baseline.',
    },
    'train-unified-conserved-sequence': {
        'module': 'dfode_kit.cli.commands.train_unified_conserved_sequence',
        'help': 'Train a shared-latent hard-conserved sequence model.',
    },
    'train-latent-sequence': {
        'module': 'dfode_kit.cli.commands.train_latent_sequence',
        'help': 'Train a minimal AE + GRU latent rollout baseline.',
    },
}


def load_command_specs():
    return OrderedDict(sorted(_COMMAND_SPECS.items(), key=lambda item: item[0]))


def load_command(command_name, command_specs=None):
    command_specs = command_specs or load_command_specs()
    if command_name not in command_specs:
        raise KeyError(command_name)

    module_name = command_specs[command_name]['module']
    return importlib.import_module(module_name)
