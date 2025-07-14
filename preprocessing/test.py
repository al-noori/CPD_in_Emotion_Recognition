from pathlib import Path
from socket import gethostname

"""
The following functions select paths based on the hostname of the system they are run on,
so that paths locally and on the server are automatically adjusted.
"""
if gethostname() in ['lewibre']:  # local
    DATA_PATH = Path('..', 'data')
    LOGS_PATH = Path('..', 'experiments')
    DATASET_PATH = Path('..', "datasets")
    MODELS_PATH = Path('..', 'experiments')
    PREDICTIONS_PATH = Path('..', 'predictions')
    SCORES_PATH = Path('..', 'scores')
    PLOTS_PATH = Path('..', 'plots')
else:  # server
    USERNAME = Path().home().parts[-1]  # executing username (server only)

    DATA_PATH = Path('/mnt', 'stud', 'work', USERNAME, 'data')
    DATASET_PATH = Path('/mnt', 'stud', 'home', USERNAME, 'push-pull-loss', 'datasets')
    LOGS_PATH = Path('/mnt', 'stud', 'work', USERNAME, 'project', 'repo', 'experiments')
    MODELS_PATH = Path('/mnt', 'stud', 'work', USERNAME, 'project', 'repo', 'experiments', 'saved_models')
    PREDICTIONS_PATH = Path('/mnt', 'work', USERNAME, 'project', 'repo', 'experiments', 'predictions')
    SCORES_PATH = Path('/mnt', 'work', USERNAME, 'project', 'repo', 'experiments', 'scores')
    PLOTS_PATH = Path('/mnt', 'work', USERNAME, 'project', 'repo', 'experiments', 'plots')


def mkdir(path):
    """ Short form of path.mkdir(parents=True, exist_ok=True) while also returning path.

    Args:
        path: pathlib.Path

    Return:
        path: pathlib.Path
    """
    path.mkdir(parents=True, exist_ok=True)
    print(path)
    return path

if name == 'main':
    print(gethostname())