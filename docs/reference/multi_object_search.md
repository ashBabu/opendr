# multi_object_search module

The *multi_object_search* module contains the *ExplorationRLLearner* class, which inherits from the abstract class *LearnerRL*.

### Class ExplorationRLLearner
Bases: `engine.learners.LearnerRL`

The *ExplorationRLLearner* class is an RL agent that can be used to train wheeled robots for combining short-horizon control with long horizon reasioning into a single policy.
Originally published in [[1]](#multi_object_search), Learning Long-Horizon Robot Exploration Strategies for Multi-Object Search in Continuous Action Spaces,(https://arxiv.org/abs/2205.11384).

The [ExplorationRLLearner](/src/opendr/control/multi_object_search/multi_object_search_learner.py) class has the following public methods:

#### `ExplorationRLLearner` constructor

Constructor parameters:

- **env**: *gym.Env*\
  Reinforcment learning environment to train or evaluate the agent on.
- **lr**: *float, default=1e-5*\
  Specifies the initial learning rate to be used during training.
- **ent_coef**: *float, default=0.005*\
  Specifies the entropy coefficient used as additional loss penalty during training.
- **clip_range**: *float, default=0.1*\
  Specifies the clipping parameter for PPO.
- **gamma**: *float, default=0.99*\
  Specifies the discount factor during training.
- **n_steps**: *int, default=2048*\
  Specifies the number of steps to run for each environment per update during training.
- **n_epochs**: *int, default=4*\
  Specifies the number of epochs when optimizing the surrogate loss during training.
- **iters**: *int, default=1_000_000*\
  Specifies the number of steps the training should run for.
- **batch_size**: *int, default=64*\
  Specifies the batch size during training.
- **lr_schedule**: *{'', 'linear'}, default='linear'*\
  Specifies the learning rate scheduler to use. Empty to use a constant rate.
- **backbone**: *{'MultiInputPolicy'}, default='MultiInputPolicy'*\
  Specifies the architecture for the RL agent.
- **checkpoint_after_iter**: *int, default=20_000*\
  Specifies per how many training steps a checkpoint should be saved. If it is set to 0 no checkpoints will be saved.
- **temp_path**: *str, default=''*\
  Specifies a path where the algorithm stores log files and saves checkpoints.
- **device**: *{'cpu', 'cuda'}, default='cuda'*\
  Specifies the device to be used.
- **seed**: *int, default=None*\
  Random seed for the agent. If None a random seed will be used.
- **nr_evaluations**: *int, default=50*\
  Number of episodes to evaluate over.

#### `ExplorationRLLearner.fit`
```python
ExplorationRLLearner.fit(self, env, logging_path, silent, verbose)
```

Train the agent on the environment.

Parameters:

- **env**: *gym.Env, default=None*\
  If specified use this env to train.
- **logging_path**: *str, default=''*\
  Path for logging and checkpointing.
- **silent**: *bool, default=False*\
  Disable verbosity.
- **verbose**: *bool, default=True*\
  Enable verbosity.


#### `ExplorationRLLearner.eval`
```python
ExplorationRLLearner.eval(self, env, name_prefix='', nr_evaluations: int = None, deterministic_policy: bool = False)
```
Evaluate the agent on the specified environment.

Parameters:

- **env**: *gym.Env, default=None*\
  Environment to evaluate on.
- **name_prefix**: *str, default=''*\
  Name prefix for all logged variables.
- **nr_evaluations**: *int, default=None*\
  Number of episodes to evaluate over.
- **deterministic_policy**: *bool, default=False*\
  Use deterministic or stochastic policy.


#### `ExplorationRLLearner.save`
```python
ExplorationRLLearner.save(self, path)
```
Saves the model in the path provided.

Parameters:

- **path**: *str*\
  Path to save the model, including the filename.


#### `ExplorationRLLearner.load`
```python
ExplorationRLLearner.load(self, path)
```
Loads a model from the path provided.

Parameters:

- **path**: *str*\
  Path of the model to be loaded.



#### Simulation Setup
The repository uses the iGibson Simulator as well as Stable-baseline3 as external libaries. 

This means that the training environment relies on running using the iGibson scenes. For that it is necessary to download the iGibson scenes. A script is provided in [multi_object_search](/src/opendr/control/multi_object_search/requirements_installations.py) 
To download he iGibson and the inflated traversability maps, please execute the following script and accept the agreement.

```sh
python requirements_installations.py
````

The iGibson dataset requires a valid license, which needs to be added manually. The corresponding link can be found here https://docs.google.com/forms/d/e/1FAIpQLScPwhlUcHu_mwBqq5kQzT2VRIRwg_rJvF0IWYBk_LxEZiJIFg/viewform.
For more information please have a look on the official website: https://stanfordvl.github.io/iGibson/dataset.html

##### Visualisation
For visualizating the egocentric maps and their corresponding static map, add the flag `show_map=true` in`config.yaml`.


#### Examples
* **Training and evaluation in the iGibson environment on a Multi Object Task.**.
As described above, follow the download instructions.
  ```python
    import torch
    from opendr.control.multi_object_search import ExplorationRLLearner 
    from opendr.control.multi_object_search import MultiObjectEnv
    from pathlib import Path
    from igibson.utils.utils import parse_config


    main_path = Path(__file__).parent
    logpath = f"{main_path}/logs/demo_run"
    CONFIG_FILE = str(f"{main_path}/best_defaults.yaml")
    

    env = MultiObjectEnv(config_file=CONFIG_FILE, scene_id="Benevolence_1_int")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = parse_config(CONFIG_FILE)

    agent = ExplorationRLLearner(env, device=device, iters=config.get('train_iterations', 500),temp_path=logpath,config_filename=CONFIG_FILE)

    # start training
    agent.fit(env, val_env=eval_env)

    # evaluate on finding 6 objects on one test scene
    metrics = agent.eval(env,name_prefix='Multi_Object_Search', name_scene="Benevolence_1_int", nr_evaluations= 75,deterministic_policy = False)

    print(f"Success-rate for {scene} : {metrics['metrics']['success']} \nSPL for {scene} : {metrics['metrics']['spl']}")

    
  ```

* **Evaluate a pretrained model**.
  
  ```python
    import torch
    from opendr.control.multi_object_search import ExplorationRLLearner 
    from opendr.control.multi_object_search import MultiObjectEnv
    from pathlib import Path
    from igibson.utils.utils import parse_config


    main_path = Path(__file__).parent
    logpath = f"{main_path}/logs/demo_run"
    #best_defaults.yaml contains important settings. (see above)
    CONFIG_FILE = str(f"{main_path}/best_defaults.yaml")
    

    env = MultiObjectEnv(config_file=CONFIG_FILE, scene_id="Benevolence_1_int")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = parse_config(CONFIG_FILE)

    agent = ExplorationRLLearner(env, device=device, iters=config.get('train_iterations', 500),temp_path=logpath,config_filename=CONFIG_FILE)

    
    # evaluate on finding 6 objects on all test scenes
    eval_scenes = ['Benevolence_1_int', 'Pomaria_2_int', 'Benevolence_2_int', 'Wainscott_0_int', 'Beechwood_0_int',
                'Pomaria_1_int', 'Merom_1_int']

    agent.load("pretrained")

    deterministic_policy = config.get('deterministic_policy', False)

    for scene in eval_scenes:
      metrics = agent.eval(env,name_prefix='Multi_Object_Search', name_scene=scene, nr_evaluations= 75,\
                deterministic_policy = deterministic_policy)

      print(f"Success-rate for {scene} : {metrics['metrics']['success']} \nSPL for {scene} : {metrics['metrics']['spl']}")
  ```

#### Notes

The iGibson simulator might crash, when evaluating multiple envrionments and use the gui mode.

#### References
<a name="multi-object-search" href="https://arxiv.org/abs/2205.11384">[1]</a> Learning Long-Horizon Robot Exploration Strategies for Multi-Object Search in Continuous Action Spaces,
[arXiv](https://arxiv.org/abs/2205.11384).
