import os
import pickle
import zipfile

from flax.serialization import from_bytes, to_bytes


def save_to_zip(save_dir, agent_dict):
    """
    """
    # serialize agent components
    agent_bytes = {name: pickle.dumps(target) for
                   name, target in agent_dict.items()}

    # add extension if not specified
    if isinstance(save_dir, str):
        _, ext = os.path.splitext(save_dir)
        if not ext:
            save_dir += ".zip"

    # create the zip archive and save the agent components
    with zipfile.ZipFile(save_dir, "w") as f:
        for name, serialized in agent_bytes.items():
            f.writestr(name, serialized)


def load_from_zip(agent_dict, load_path):
    """
    """
    # Check if the file exists
    if isinstance(load_path, str):
        if not os.path.exists(load_path):
            if  os.path.exists(load_path + ".zip"):
                load_path += ".zip"
            else:
                raise ValueError("Error: the file {:} could not be found.".format(load_path))

    # Open file and load the agent components
    with zipfile.ZipFile(load_path, "r") as f:
        namelist = f.namelist()
        for name, target in agent_dict.items():
            # Skip components that were not saved
            if name not in namelist:
                continue

            serialized = f.read(name)
            agent[name] = from_bytes(target, serialized)

    return agent_dict
