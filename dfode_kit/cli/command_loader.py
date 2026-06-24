import importlib
from collections import OrderedDict


_COMMAND_SPECS = {
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
    'evaluate-latent-sequence': {
        'module': 'dfode_kit.cli.commands.evaluate_latent_sequence',
        'help': 'Evaluate a latent sequence model checkpoint.',
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
