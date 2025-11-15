# Supporting Code for Efficient Multiagent Planning via Shared Action Suggestions

This repository contains the code and supplementary materials for the paper "Efficient Multiagent Planning via Shared Action Suggestions".

[arXiv Paper](https://arxiv.org/abs/2412.11430)
<!-- [AAAI Proceedings]() -->

## Repository Structure

```
├── src/
│   ├── problems.jl                     # Defines default parameters for the problems
│   ├── simulate.jl                     # Contains code for running individual simulations
│   ├── simulation_runner.jl            # Runs all simulations
│   ├── policy_generator.jl             # Used to generate additional policies
│   ├── output_data.jl                  # Displays results in a table
│   ├── policies/                       # Contains a subset of generated policies
│   └── results/                        # Contains simulation results
│
├── MultiAgentPOMDPProblems/
│   └── src/
│       ├── BoxPush/                    # Cooperative Box Push
│       ├── JointMeet/                  # Meeting in a Grid
│       ├── broadcast_channel.jl        # Broadcast Channel
│       ├── multi_tiger_pomdp.jl        # Dec-Tiger
│       ├── multi_wireless.jl           # Wireless Network
│       ├── sotchastic_mars.jl          # Mars Rover
│       ├── vis_utils.jl                # Visualization Utilities
│       └── MultiAgentPOMDPProblems.jl  # Package definition
|
├── install.jl                        # Installs the package
├── readme.md                         # This file
└── Project.toml                      # Defines the package requirements
```

## Installation

All of the developement was done with Julia 1.10 and Julia 1.11. We recommend using the most up to date stable release, but it should work with Julia 1.8 or later. You can download Julia [here](https://julialang.org/downloads/).

### Cloning the Repository

This repository uses a git submodule for the `MultiAgentPOMDPProblems` package. To clone the repository with the submodule, use:

```bash
git clone --recurse-submodules https://github.com/sisl/MCAS.git
```

If you've already cloned the repository without the submodule, initialize it with:

```bash
git submodule update --init --recursive
```

### Setting Up the Julia Environment

To setup your environment, run the following from the root directory:

```julia
julia> include("install.jl")
```

or manually:

```julia
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.instantiate()
julia> Pkg.develop(path="./MultiAgentPOMDPProblems")
julia> Pkg.precompile()
```


## Problem Implementation Details

For details on the implemenation of the problem used, reference the `MultiAgentPOMDPProblems` package. The parameters used for each problem can be found in the `problems.jl` file.


## Generating Policies

Only a subset of policies are included due to memory constraints. To generate additional policies, use `src/policy_generator.jl`. The parameters used to generate the policies are in `PROBLEMS_TO_RUN` with only the time limit being changed from the SARSOP parameters. Most policices converged prior to the time limit.

## Running Simulations

To run single simulations and visualize how the belief changes with the sharing of suggestions, use `src/simulate.jl`.  

### Example Usage

```julia
using POMDPs
using POMDPTools
using MultiAgentPOMDPProblems
using ProgressMeter
using Printf
using JLD2

using DiscreteValueIteration # needed for loading the MMDP policy

# Used for handling results
using CSV
using Logging

include("src/problems.jl")
include("src/suggested_action_policies.jl")
include("src/simulate.jl")
include("src/problem_and_policy_helpers.jl")


problem_symbol = :tiger_3
control_option = :conflate_action

joint_problem, agent_problems, joint_policy, agent_policies, joint_mdp_policy = load_policy(problem_symbol)

joint_control = get_controller(:mpomdp, joint_problem, joint_policy, agent_problems, agent_policies) # Used to visualize the joint belief for comparison
# joint_control = nothing # This should be used when not wanting a visualization of the joint policy
control = get_controller(control_option, joint_problem, joint_policy, agent_problems, agent_policies; delta_single=1e-5, delta_joint=1e-5, max_beliefs=200)

seed = 42

s0 = rand(MersenneTwister(seed), initialstate(joint_problem))

results = run_simulation(
    joint_problem, s0, control;
    seed=seed,
    text_output=false,
    max_steps=10,
    show_plots=true,
    joint_control=joint_control
)
```

### Example Usage (Meeting in a 2 x 2 Grid UI, WP)


```julia
# Same packages as above and included files

problem_symbol = :joint_meet_2x2_ui_wp
control_option = :conflate_action

joint_problem, agent_problems, joint_policy, agent_policies, joint_mdp_policy = load_policy(problem_symbol)

joint_control = get_controller(:mpomdp, joint_problem, joint_policy, agent_problems, agent_policies) # Used to visualize the joint belief for comparison
# joint_control = nothing # This should be used when not wanting a visualization of the joint policy
control = get_controller(control_option, joint_problem, joint_policy, agent_problems, agent_policies; delta_single=1e-5, delta_joint=1e-5, max_beliefs=200)

seed = 43

s0 = rand(MersenneTwister(seed), initialstate(joint_problem))

results = run_simulation(
    joint_problem, s0, control;
    seed=seed,
    text_output=false,
    max_steps=10,
    show_plots=true,
    joint_control=joint_control
)
```

## Paper Experiments

To recreate all experiments in the paper, run `src/simulation_runner.jl`. This requires policies for all of the problems and must be computed using `src/policy_generator.jl`.


The csv for the experiments is located at `src/results/results_2024-10-15.csv`. To view the results in a table format, run `src/output_data.jl`.

## Citation

```bib
@article{asmar2026mcas,
  title     = {Efficient Multiagent Planning via Shared Action Suggestions},
  author    = {Dylan M. Asmar and Mykel J. Kochenderfer},
  booktitle = {AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```