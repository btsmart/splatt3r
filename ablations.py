from main import *


def default_run():

    # Setup the workspace (eg. load the config, create a directory for results at config.save_dir, etc.)
    config_location = "configs/main.yaml"
    config = workspace.load_config(config_location, None)
    if os.getenv("LOCAL_RANK", '0') == '0':
        config = workspace.create_workspace(config)

    # Run the experiment
    run_experiment(config)


def with_mast3r_loss():

    # Setup the workspace (eg. load the config, create a directory for results at config.save_dir, etc.)
    config_location = "configs/with_mast3r_loss.yaml"
    config = workspace.load_config(config_location, None)
    if os.getenv("LOCAL_RANK", '0') == '0':
        config = workspace.create_workspace(config)

    # Run the experiment
    run_experiment(config)


def without_masking():

    # Setup the workspace (eg. load the config, create a directory for results at config.save_dir, etc.)
    config_location = "configs/without_masking.yaml"
    config = workspace.load_config(config_location, None)
    if os.getenv("LOCAL_RANK", '0') == '0':
        config = workspace.create_workspace(config)

    # Run the experiment
    run_experiment(config)


def without_lpips_loss():

    # Setup the workspace (eg. load the config, create a directory for results at config.save_dir, etc.)
    config_location = "configs/without_lpips_loss.yaml"
    config = workspace.load_config(config_location, None)
    if os.getenv("LOCAL_RANK", '0') == '0':
        config = workspace.create_workspace(config)

    # Run the experiment
    run_experiment(config)


def without_offset():

    # Setup the workspace (eg. load the config, create a directory for results at config.save_dir, etc.)
    config_location = "configs/without_offset.yaml"
    config = workspace.load_config(config_location, None)
    if os.getenv("LOCAL_RANK", '0') == '0':
        config = workspace.create_workspace(config)

    # Run the experiment
    run_experiment(config)


if __name__ == "__main__":

    # Somewhat hacky way to fetch the function corresponding to the ablation we want to run
    ablation_name = sys.argv[1]
    ablation_function = locals().get(ablation_name)

    # Run the ablation if it exists
    if ablation_function:
        ablation_function()
    else:
        raise NotImplementedError(
            f"Ablation name '{sys.argv[1]}' not recognised")
