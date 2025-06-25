function load_policy(problem_symbol::Symbol)
    # Load joint_problem, agent_problems, joint_policy, agent_policies, joint_mdp_policy
    load_path = joinpath("src", "policies", "$problem_symbol.jld2")
    
    if !isfile(load_path)
        error("Policy file not found: $load_path. Reference policy_generator.jl to generate policies.")
    end
    
    loaded_data = JLD2.load(load_path)
    joint_problem = loaded_data["joint_problem"]
    agent_problems = loaded_data["agent_problems"]
    joint_policy = loaded_data["joint_policy"]
    agent_policies = loaded_data["agent_policies"]
    joint_mdp_policy = loaded_data["joint_mdp_policy"]

    return joint_problem, agent_problems, joint_policy, agent_policies, joint_mdp_policy
end

function get_controller(
    control_option::Symbol, joint_problem, joint_policy, agent_problems, agent_policies;
    delta_single::Float64=1e-5,
    delta_joint::Float64=1e-5,
    max_beliefs::Int=1_000_000
)
    if control_option == :mpomdp
        control = JointPolicy(joint_problem, joint_policy)
    elseif control_option == :pomdp_1
        control = SinglePolicy(agent_problems[1], 1, agent_policies[1])
    elseif control_option == :pomdp_2
        control = SinglePolicy(agent_problems[2], 2, agent_policies[2])
    elseif control_option == :independent
        control = Independent(agent_problems, agent_policies)
    elseif control_option == :conflate_joint
        control = ConflateJoint(joint_problem, joint_policy, agent_problems, agent_policies)
    elseif control_option == :conflate_alpha
        control = Conflation(joint_problem, joint_policy, agent_problems, agent_policies;
            prune_option=:alpha,
            joint_belief_delta=delta_joint,
            single_belief_delta=delta_single,
            max_surrogate_beliefs=max_beliefs
        )
    elseif control_option == :conflate_action
        control = Conflation(joint_problem, joint_policy, agent_problems, agent_policies;
            prune_option=:action,
            joint_belief_delta=delta_joint,
            single_belief_delta=delta_single,
            max_surrogate_beliefs=max_beliefs
        )
    elseif control_option == :conflate_action_expected
        control = Conflation(joint_problem, joint_policy, agent_problems, agent_policies;
            prune_option=:action,
            joint_belief_delta=delta_joint,
            single_belief_delta=delta_single,
            max_surrogate_beliefs=max_beliefs,
            selection_option=:expected_value
        )
    else
        throw(ArgumentError("Invalid control option: $control_option"))
    end
end

function print_policy_values(problem_symbol::Symbol)
    p = get_problem(problem_symbol, 1)
    num_agents = p.num_agents
    
    # Load the CSV file
    csv_file = joinpath("src", "policies", "policy_values.csv")
    df = CSV.read(csv_file, DataFrame)

    # Filter for the most recent entries for the given problem symbol
    most_recent_entries = df[df[:, :problem] .== string(problem_symbol), :]

    # Sort by date and time to get the most recent ones first
    most_recent_entries = sort(most_recent_entries, [:date, :time], rev=true)

    # Get the most recent entry for each policy
    unique_policies = unique(most_recent_entries[:, :policy])
    most_recent_policy_entries = DataFrame()

    for policy in unique_policies
        policy_entries = most_recent_entries[most_recent_entries[:, :policy] .== policy, :]
        most_recent_policy_entry = first(policy_entries, 1)
        append!(most_recent_policy_entries, most_recent_policy_entry)
    end

    # Create a dictionary mapping policy to value for the most recent entries
    policy_to_value = Dict(most_recent_policy_entries[:, :policy] .=> most_recent_policy_entries[:, :value])

    @printf("%-10s : %10.4f\n", "MMDP", policy_to_value["mmdp"])
    @printf("%-10s : %10.4f\n", "MPOMDP", policy_to_value["mpomdp"])
    for ii in 1:num_agents
        pomdp_str = "pomdp_$ii"
        @printf("%-10s : %10.4f\n", uppercase(pomdp_str), policy_to_value[pomdp_str])
    end
end
