
import POMDPTools: action_info
import POMDPs: action

abstract type MultiAgentControlStrategy <: Policy end

function action_info(control::MultiAgentControlStrategy)
    error("`action_info` is not defined for type $(typeof(control)).")
end
function update_belief!(control::MultiAgentControlStrategy, act, o)
    error("`update_belief!` is not defined for type $(typeof(control)).")
end
function get_observation(control::MultiAgentControlStrategy, o::O) where {O}
    error("`get_observation` is not defined for type $(typeof(control)).")
end

mutable struct JointPolicy{S, A, O} <: MultiAgentControlStrategy
    const problem::POMDP{S, A, O}
    const policy::Policy
    const updater
    belief
end
function JointPolicy(problem::POMDP, policy::Policy)
    up = updater(policy)
    b = initialize_belief(up, initialstate(problem))
    return JointPolicy(problem, policy, up, b)
end

"""
    SinglePolicy

When using a single policy in a multi-agent setting. E.g. a joint policy, or a single agent
    that is making decisions for all agents based on it's own belief.
"""
mutable struct SinglePolicy{S, A, O} <: MultiAgentControlStrategy
    const problem::POMDP{S, A, O}
    const agent_idx::Int
    const policy::AlphaVectorPolicy
    const updater::DiscreteUpdater
    belief
end
function SinglePolicy(problem::POMDP, agent_idx::Int, policy::AlphaVectorPolicy)
    up = updater(policy)
    b = initialize_belief(up, initialstate(problem))
    return SinglePolicy(problem, agent_idx, policy, up, b)
end

"""
    Independent

When each agent has its own problem (model of world), policy, updater, and belief.
"""
struct Independent{S, A, O} <: MultiAgentControlStrategy
    indiv_control::Vector{SinglePolicy{S, A, O}}
end
function Independent(problems::Vector{<:POMDP}, policies::Vector{AlphaVectorPolicy})
    indiv_control = [SinglePolicy(p, ii, pol) for (ii, (p, pol)) in enumerate(zip(problems, policies))]
    return Independent(indiv_control)
end

"""
    ConflateJoint

Agents maintain their own beliefs, and select from the joint policy based on conflating 
    their beliefs. This is simlar to forming a joint belief.
"""
struct ConflateJoint{S, A, O} <: MultiAgentControlStrategy
    joint_problem::POMDP{S, A, O}
    joint_policy::AlphaVectorPolicy
    indiv_control::Vector{SinglePolicy{S, A, O}}
end
function ConflateJoint(
    joint_problem::POMDP,
    joint_policy::AlphaVectorPolicy,
    problems::Vector{<:POMDP},
    policies::Vector{AlphaVectorPolicy}
)
    indiv_control = [SinglePolicy(p, ii, pol) for (ii, (p, pol)) in enumerate(zip(problems, policies))]
    return ConflateJoint(joint_problem, joint_policy, indiv_control)
end

"""
    EstimatedBelief
    
The estimated belief set for an agent.
"""
mutable struct EstimatedBelief
    const problem::POMDP
    const policy::AlphaVectorPolicy
    const updater::DiscreteUpdater
    beliefs::Vector
    weights::Vector{Float64}
end
function EstimatedBelief(problem::POMDP, policy::AlphaVectorPolicy)
    up = updater(policy)
    b = initialize_belief(up, initialstate(problem))
    return EstimatedBelief(problem, policy, up, [b], [1.0])
end

Base.length(eb::EstimatedBelief) = length(eb.beliefs)

"""
    Conflation

MCAS using clongation to combine the beliefs.
"""
mutable struct Conflation{S, A, O} <: MultiAgentControlStrategy
    const joint_problem::POMDP{S, A, O}
    const joint_policy::AlphaVectorPolicy
    const indiv_control::Vector{SinglePolicy{S, A, O}}
    const surrogate_beliefs::Dict{Int, EstimatedBelief}
    const prune_option::Symbol
    const joint_belief_delta::Float64
    const single_belief_delta::Float64
    const max_surrogate_beliefs::Dict{Int, Int}
    const selection_option::Symbol
    const weight_option::Symbol
    prev_selected_belief::Union{Nothing, DiscreteBelief, SparseCat, Vector{Float64}}
end
function Conflation(
    joint_problem::POMDP{S, A, O},
    joint_policy::AlphaVectorPolicy,
    problems::Vector{<:POMDP},
    policies::Vector{AlphaVectorPolicy};
    prune_option::Symbol=:alpha,
    joint_belief_delta::Float64=0.0,
    single_belief_delta::Float64=0.0,
    max_surrogate_beliefs::Union{Int, Dict{Int, Int}}=1_000_000,
    selection_option::Symbol=:max_weight,
    weight_option::Symbol=:count
) where {S,A,O}
        
    num_agents = problems[1].num_agents
    
    if typeof(max_surrogate_beliefs) == Int
        max_surrogate_beliefs = Dict(ii => max_surrogate_beliefs for ii in 2:num_agents)
    end
    
    indiv_control = Vector{SinglePolicy{S,A,O}}(undef, num_agents)
    surrogate_beliefs = Dict{Int, EstimatedBelief}()
    for ii in 1:num_agents
        indiv_control[ii] = SinglePolicy(problems[ii], ii, policies[ii])
        if ii != 1
            # Surrogate beliefs are using the actual problems and policies of other agents
            surrogate_beliefs[ii] = EstimatedBelief(problems[ii], policies[ii])
        end
    end
    return Conflation(joint_problem, joint_policy, indiv_control, 
        surrogate_beliefs, prune_option, joint_belief_delta, single_belief_delta, 
        max_surrogate_beliefs, selection_option, weight_option, nothing)
end

# Extended to get alpha vector index back in the info NamedTuple
function POMDPTools.action_info(p::AlphaVectorPolicy, b)
    bvec = beliefvec(p.pomdp, p.n_states, b)
    num_vectors = length(p.alphas)
    best_idx = 1
    max_value = -Inf
    for i = 1:num_vectors
        temp_value = dot(bvec, p.alphas[i])
        if temp_value > max_value
            max_value = temp_value
            best_idx = i
        end
    end
    info = (idx=best_idx,)
    return p.action_map[best_idx], info
end

function action(control::MultiAgentControlStrategy)
    act, _ = action_info(control)
    return act
end

function action_info(control::SinglePolicy)
    return POMDPTools.action_info(control.policy, control.belief)
end

function action_info(control::JointPolicy)
    return POMDPTools.action_info(control.policy, control.belief)
end

function action_info(control::Independent)
    acts, infos = [], []
    for cp in control.indiv_control
        act, info = action_info(cp)
        push!(acts, act)
        push!(infos, info)
    end
    
    # Combine infos into a single NamedTuple
    info_keys = Tuple(Symbol("agent_$ii") for ii in 1:length(infos))
    joint_info = NamedTuple{info_keys}(infos)
    # Joint action is the individual actions for each agent
    joint_act = [ai[idx] for (idx, ai) in enumerate(acts)]
    return joint_act, joint_info
end

function action_info(control::ConflateJoint)
    # Get conflated belief of all agents
    joint_conflated_b = conflate_beliefs(control)
    return POMDPTools.action_info(control.joint_policy, joint_conflated_b)
end

function action_info(control::Conflation{S, A, O}) where {S,A,O}
    num_agents = control.joint_problem.num_agents
    # 1. Determine individual actions for agents 2:num_agents
    acts = Vector{A}(undef, num_agents - 1)
    alpha_idxs = Vector{Int}(undef, num_agents - 1)
    for ii in 2:num_agents
        ic = control.indiv_control[ii]
        a_i, info_i = action_info(ic.policy, ic.belief)
        acts[ii-1] = a_i
        alpha_idxs[ii-1] = info_i.idx
    end
    
    # 2. Prune the surrogate beliefs based on shared action or alpha vector
    if control.prune_option == :action
        prune!(control, acts)
    elseif control.prune_option == :alpha
        prune!(control, alpha_idxs)
    else
        throw(ArgumentError("Invalid options of `prune_option`: $(control.prune_option)"))
    end
    
    for ii in 2:num_agents
        normalize!(control.surrogate_beliefs[ii].weights, 1)
    end
    
    # 3. Estimate joint belief from remaining beliefs
    est_joint_b = select_belief(control)
    control.prev_selected_belief = est_joint_b

    return POMDPTools.action_info(control.joint_policy, est_joint_b)
end

function prune!(control::Conflation{S, A, O}, acts::Vector{A}) where {S,A,O}
    num_agents = control.joint_problem.num_agents
    for ii in 2:num_agents
        eb = control.surrogate_beliefs[ii]
        delete_idxs = Int[]
        one_remaining = false
        for (jj, b_j) in enumerate(eb.beliefs)
            a_j = action(eb.policy, b_j)
            if a_j != acts[ii - 1]
                push!(delete_idxs, jj)
            else
                one_remaining = true
            end
        end
        if one_remaining
            unique!(sort!(delete_idxs))
            deleteat!(eb.beliefs, delete_idxs)
            deleteat!(eb.weights, delete_idxs)
        else
            @debug "Shared action would have pruned all surrogate beliefs for Agent $ii"
        end
        enforce_max_size!(eb, control.max_surrogate_beliefs[ii])
    end
end

function prune!(control::Conflation, alpha_idx::Vector{Int})
    num_agents = control.joint_problem.num_agents
    for ii in 2:num_agents
        eb = control.surrogate_beliefs[ii]
        delete_idxs = Int[]
        one_remaining = false
        for (jj, b_j) in enumerate(eb.beliefs)
            a_j, info_j = action_info(eb.policy, b_j)
            alpha_idx_j = info_j.idx
            if alpha_idx_j != alpha_idx[ii - 1]
                push!(delete_idxs, jj)
            else
                one_remaining = true
            end
        end
        if one_remaining
            unique!(sort!(delete_idxs))
            deleteat!(eb.beliefs, delete_idxs)
            deleteat!(eb.weights, delete_idxs)
        else
            @debug "Shared alpha vector index would have pruned all surrogate beliefs for Agent $ii"
        end
        enforce_max_size!(eb, control.max_surrogate_beliefs[ii])
    end
end

function enforce_max_size!(eb::EstimatedBelief, max_size::Int)
    beliefs = eb.beliefs
    weights = eb.weights
    # If number of beleifs is greater than the max size, reduce the number by removing
    # similar beliefs even if outside of delta.
    if length(beliefs) > max_size
        @debug "Reached max surrogate beliefs: $(length(beliefs)), reducing to $max_size"
        
        n = length(beliefs)
        dists = Vector{NamedTuple}(undef, div(n * (n - 1), 2))
        idx = 1
        for idx_i in 1:(n-1)
            for idx_j in (idx_i+1):length(beliefs)
                dists[idx] = (distance=norm(beliefs[idx_i].b - beliefs[idx_j].b, 1), idx_i=idx_i, idx_j=idx_j)
                idx += 1
            end
        end

        # Sort distances in ascending order
        sort!(dists, by=x->x.distance)

        # Initialize a boolean array to track beliefs to keep
        keep = trues(n)
        current_entry = 1

        # Remove beliefs until the desired number is reached
        while sum(keep) > max_size && current_entry <= length(dists)
            idx_i = dists[current_entry].idx_i
            idx_j = dists[current_entry].idx_j

            if keep[idx_i] && keep[idx_j]
                # Remove the belief with the lower weight
                if weights[idx_i] <= weights[idx_j]
                    keep[idx_i] = false
                    weights[idx_j] += weights[idx_i]
                else
                    keep[idx_j] = false
                    weights[idx_i] += weights[idx_j]
                end
            end
            current_entry += 1
        end

        if sum(keep) > max_size
            error("Shouldn't be able to get here. Check the logic above.")
        end

        # Update beliefs and weights to only include kept beliefs
        eb.beliefs = eb.beliefs[keep]
        eb.weights = eb.weights[keep]
    end
end

#* Important function!!
function select_belief(control::Conflation)
    num_agents = control.joint_problem.num_agents
    
    # Combine all possible options for the belief and select from the conflations
    num_possible_beliefs = prod(length(control.surrogate_beliefs[ii]) for ii in 2:num_agents)

    conflated_beliefs = Vector{DiscreteBelief}()
    weights = Float64[]

    vec_of_vec_of_idxs = [collect(1:length(control.surrogate_beliefs[ii])) for ii in 2:num_agents]
    
    vec_of_vec_of_beliefs = [control.surrogate_beliefs[ii].beliefs for ii in 2:num_agents]
    vec_of_vec_of_weights = [normalize(control.surrogate_beliefs[ii].weights, 1) for ii in 2:num_agents]

    idx_combinations = collect(Iterators.product(vec_of_vec_of_idxs...))
    belief_combinations = collect(Iterators.product(vec_of_vec_of_beliefs...))
    weight_combinations = collect(Iterators.product(vec_of_vec_of_weights...))

    # Vector of vector to track indexs that are not orthogonal for some combination
    belief_not_orthogonal = [falses(length(v)) for v in vec_of_vec_of_idxs]
    
    @assert length(idx_combinations) == num_possible_beliefs "Number of beliefs is not correct."

    for ii in 1:num_possible_beliefs
        indiv_beliefs = vcat(control.indiv_control[1].belief, collect(belief_combinations[ii]))
        
        try
            push!(conflated_beliefs, conflate_beliefs(control, indiv_beliefs))
            push!(weights, prod(weight_combinations[ii]))
            idxs = idx_combinations[ii]
            for jj in 1:(num_agents - 1)
               belief_not_orthogonal[jj][idxs[jj]] = true 
            end
        catch e
            if isa(e, ErrorException) && occursin("Beliefs are orthogonal", e.msg)
                continue
            else
                rethrow(e)
            end
        end
    end
    if isempty(conflated_beliefs)
        error("All surrogate beliefs are orthogonal to agent's belief")
    else
        # delete any beliefs that were orthogonal for all combinations (false in the 
        # belief_not_orthogonal vector)
        for ii in 2:num_agents
            # Find any falses
            delete_idxs = findall(x -> !x, belief_not_orthogonal[ii-1])
            unique!(sort!(delete_idxs))
            deleteat!(control.surrogate_beliefs[ii].beliefs, delete_idxs)
            deleteat!(control.surrogate_beliefs[ii].weights, delete_idxs)
        end
    end
    
    # Prune conflated beliefs if the joint beliefs are same/close (L1 within some delta) or
    # if the conflated beliefs fall under the same dominated belief subspace of the joint
    # policy. The in second case, we will be picking the same action, so we don't need to
    # reason over all of them.
    delete_idxs = Int[]
    for ii in 1:(length(conflated_beliefs) - 1)
        if ii in delete_idxs
            continue
        end
        cb_i = conflated_beliefs[ii]
        a_i, info_i = action_info(control.joint_policy, cb_i)
        alpha_idx_i = info_i.idx
        for jj in (ii + 1):length(conflated_beliefs)
            if jj in delete_idxs
                continue
            end
            cb_j = conflated_beliefs[jj]
            a_j, info_j = action_info(control.joint_policy, cb_j)
            alpha_idx_j = info_j.idx
            if (a_i == a_j) || (norm(cb_i.b - cb_j.b, 1) <= control.joint_belief_delta)
                push!(delete_idxs, jj)
                weights[ii] += weights[jj]
            end
        end 
    end
    
    unique!(sort!(delete_idxs))
    deleteat!(conflated_beliefs, delete_idxs)
    deleteat!(weights, delete_idxs)
    
    
    #? Select the belief with the largest weight or expected value.
    #? Other ideas might be better here?
    
    if control.selection_option == :max_weight
        return conflated_beliefs[argmax(weights)]
    elseif control.selection_option == :expected_value
        value_of_cbs = Vector{Float64}(undef, length(conflated_beliefs))
        for ii in 1:length(conflated_beliefs)
            value_of_cbs[ii] = value(control.joint_policy, conflated_beliefs[ii])
        end
        expected_value_of_cbs = value_of_cbs .* weights    
        return conflated_beliefs[argmax(expected_value_of_cbs)]
    elseif control.selection_option == :min_entropy
        entropy_of_cbs = Vector{Float64}(undef, length(conflated_beliefs))
        for ii in 1:length(conflated_beliefs)
            entropy_of_cbs[ii] = entropy(conflated_beliefs[ii])
        end
        return conflated_beliefs[argmin(entropy_of_cbs)]
    elseif control.selection_option == :max_entropy
        entropy_of_cbs = Vector{Float64}(undef, length(conflated_beliefs))
        for ii in 1:length(conflated_beliefs)
            entropy_of_cbs[ii] = entropy(conflated_beliefs[ii])
        end
        return conflated_beliefs[argmax(entropy_of_cbs)]
    elseif control.selection_option == :min_l1_dist
        if isnothing(control.prev_selected_belief)
            return conflated_beliefs[argmax(weights)]
        end
        l1_dists = Vector{Float64}(undef, length(conflated_beliefs))
        for ii in 1:length(conflated_beliefs)
            l1_dists[ii] = norm(conflated_beliefs[ii].b - control.prev_selected_belief.b, 1)
        end
        return conflated_beliefs[argmin(l1_dists)]
    else
        throw(ArgumentError("Invalid selection option: $(control.selection_option)"))
    end
end

entropy(b::DiscreteBelief) = entropy(b.b)
function entropy(b::Vector{Float64})
    entropy = 0.0
    for ii in 1:length(b)
        if b[ii] > 0.0
            entropy -= b[ii] * log(b[ii])
        end
    end
    return entropy
end
function entropy(b::SparseCat)
    entropy = 0.0
    for p in b.probs
        if p > 0.0
            entropy -= p * log(p)
        end
    end
    return entropy
end


function conflate_beliefs(control::ConflateJoint)
    conflate_beliefs(control.joint_problem, [cp.belief for cp in control.indiv_control])
end
function conflate_beliefs(control::ConflateJoint, beliefs::Vector)
    conflate_beliefs(control.joint_problem, beliefs)
end
function conflate_beliefs(control::Conflation, beliefs::Vector)
    conflate_beliefs(control.joint_problem, beliefs)
end

function conflate_beliefs(pomdp::POMDP, beliefs::Vector)
    # Component product of the beliefs
    num_states = length(states(pomdp))
    bc = ones(num_states)
    for b in beliefs
        b_vec = beliefvec(pomdp, num_states, b)
        bc .*= b_vec
    end
    sum_bc = sum(bc)
    if sum_bc > 0.0
        bc = bc ./ sum_bc
    else
       throw(ErrorException("Beliefs are orthogonal"))
    end
    return DiscreteBelief(pomdp, bc)
end


function get_observation(control::SinglePolicy, o::O) where {O}
    if control.agent_idx == 0
        return o # Joint
    else
        return [o[control.agent_idx]]
    end
end

function update_belief!(control::JointPolicy{S, A, O}, act::A, o::O) where {S, A, O}
    control.belief = update(control.updater, control.belief, act, o)
    return control
end

function update_belief!(control::SinglePolicy{S, A, O}, act::A, o::O) where {S, A, O}
    control.belief = update(control.updater, control.belief, act, get_observation(control, o))
    return control
end

function update_belief!(control::Independent{S, A, O}, act::A, o::O) where {S, A, O}
    for cp in control.indiv_control
        update_belief!(cp, act, o)
    end
    return control
end

function update_belief!(control::ConflateJoint{S, A, O}, act::A, o::O) where {S, A, O}
    for cp in control.indiv_control
        update_belief!(cp, act, o)
    end
    return control
end


function update_belief!(control::Conflation{S, A, O}, act::A, o::O) where {S, A, O}
    # Update the individual agent's beliefs 
    for cp in control.indiv_control
        update_belief!(cp, act, o)
    end
    
    # Update/expand the surrogate beliefs (have to consider all possible observations)
    for ii in 2:control.joint_problem.num_agents
        eb = control.surrogate_beliefs[ii]
        new_beliefs = Vector{DiscreteBelief}()
        new_weights = Vector{Float64}()
        obs = observations(eb.problem)
        for (ii, b_i) in enumerate(eb.beliefs)
            for o_j in obs
                try
                    if control.weight_option == :count
                        p_o_b = 1.0
                    elseif control.weight_option == :observation_probability
                        p_o_b = observation_probability(eb.problem, b_i, act, o_j)
                    else
                        throw(ArgumentError("Invalid weight option: $(control.weight_option)"))
                    end
                    
                    if p_o_b > 0.0
                        bp = update(eb.updater, b_i, act, o_j)
                        if !isempty(new_beliefs)
                            dists = [norm(bp.b - new_b.b, 1) for new_b in new_beliefs]
                            min_l1_dist, min_idx = findmin(dists)
                        else
                            min_l1_dist = Inf
                        end
                        
                        if min_l1_dist <= control.single_belief_delta
                            new_weights[min_idx] += eb.weights[ii] * p_o_b
                        else
                            push!(new_beliefs, bp)
                            push!(new_weights, eb.weights[ii] * p_o_b)
                        end
                        
                    end
                catch e
                    if isa(e, ErrorException) && occursin("Failed discrete belief update: new probabilities sum to zero.", e.msg)
                        continue
                    else
                        rethrow(e)
                    end
                end
            end
        end
        eb.beliefs = new_beliefs
        eb.weights = normalize!(new_weights, 1)
    end
end

function observation_probability(pomdp::POMDP, belief::DiscreteBelief, a, o)
    obs_prob = 0.0
    states_list = belief.state_list
    for sp in states_list  # sp: next state
        od = observation(pomdp, a, sp)
        p_o_a_sp = pdf(od, o)
        if p_o_a_sp > 0.0
            for (ii, s) in enumerate(states_list)
                if belief.b[ii] > 0.0
                    td = transition(pomdp, s, a)
                    p_sp_s_a = pdf(td, sp)
                    obs_prob += p_o_a_sp * p_sp_s_a * belief.b[ii]
                end
            end
        end
    end
    return obs_prob
end

function POMDPTools.render(control::MultiAgentControlStrategy, step::NamedTuple)
    error("`render` is not defined for type $(typeof(control)).")
end

function POMDPTools.render(control::JointPolicy, plot_step::NamedTuple; title_str="")
    plot_step_ = (s=plot_step.s, a=plot_step.a, b=control.belief)
    if isempty(title_str)
        title_str = "\nCentralized"
    end
    plt = POMDPTools.render(control.problem, plot_step_; title_str=title_str)
    return plt
end

function POMDPTools.render(control::SinglePolicy, plot_step::NamedTuple; title_str="")
    plts = []
    num_plots_w = 1
    if !isnothing(get(plot_step, :joint_b, nothing))
        if !isnothing(get(plot_step, :joint_a, nothing))
            plot_step_joint = (s=plot_step.s, a=plot_step.joint_a, b=plot_step.joint_b)
        else
            plot_step_joint = (s=plot_step.s, b=plot_step.joint_b)
        end
        plt_joint = POMDPTools.render(control.problem, plot_step_joint; title_str="\nJoint Belief")
        push!(plts, plt_joint)
        num_plots_w = 2
    end
    
    plot_step_ = (s=plot_step.s, a=plot_step.a, b=control.belief)
    if isempty(title_str)
        title_str = "\nAgent $(control.agent_idx)"
    end
    plt = POMDPTools.render(control.problem, plot_step_; title_str=title_str)
    push!(plts, plt)
    w, h = plts[1].attr[:size]
    
    num_plots_h = div(length(plts), num_plots_w)
    adjusted_w = w * num_plots_w
    adjusted_h = h * num_plots_h * 1.0
    return plot(plts..., layout=(num_plots_h, num_plots_w), size=(adjusted_w, adjusted_h))
end

function POMDPTools.render(control::Independent, plot_step::NamedTuple)
    plts = []
    num_plots_w = length(control.indiv_control)
    
    if !isnothing(get(plot_step, :joint_b, nothing))
        if !isnothing(get(plot_step, :joint_a, nothing))
            plot_step_joint = (s=plot_step.s, a=plot_step.joint_a, b=plot_step.joint_b)
        else
            plot_step_joint = (s=plot_step.s, b=plot_step.joint_b)
        end
        plt_joint = POMDPTools.render(control.indiv_control[1].problem, plot_step_joint; title_str="\nJoint Belief")
        push!(plts, plt_joint)
        
        w, h = plts[1].attr[:size]
        blank_plt = plot(; legend=false, ticks=false, showaxis=false, grid=false, 
            aspectratio=:equal, size=(w, h))
        while length(plts) < num_plots_w
            push!(plts, blank_plt)
        end
    end
    
    for (i, cp) in enumerate(control.indiv_control)
        ai = action(cp)
        plot_step_i = (s=plot_step.s, a=ai, b=cp.belief)
        plt = POMDPTools.render(cp, plot_step_i; title_str="\nAgent $i")
        push!(plts, plt)
    end
    w, h = plts[1].attr[:size]
    
    num_plots_h = div(length(plts), num_plots_w)
    adjusted_w = w * num_plots_w 
    adjusted_h = h * num_plots_h * 1.0
    return plot(plts..., layout=(num_plots_h, num_plots_w), size=(adjusted_w, adjusted_h))
end

function POMDPTools.render(control::ConflateJoint, plot_step::NamedTuple)
    num_plots_w = control.joint_problem.num_agents
    plts = []
    
    if !isnothing(get(plot_step, :joint_b, nothing))
        if !isnothing(get(plot_step, :joint_a, nothing))
            plot_step_joint = (s=plot_step.s, a=plot_step.joint_a, b=plot_step.joint_b)
        else
            plot_step_joint = (s=plot_step.s, b=plot_step.joint_b)
        end
        plt_joint = POMDPTools.render(control.indiv_control[1].problem, plot_step_joint; title_str="\nJoint Belief")
        # plt_joint = plot!(plt_joint, title="Joint Belief", titlefont=font(30))
        # plt_joint = add_title!(plt_joint, "Joint Belief")
        push!(plts, plt_joint)
    end
    
    joint_conflated_b = conflate_beliefs(control)
    plot_step_joint = (s=plot_step.s, a=plot_step.a, b=joint_conflated_b)
    plt = POMDPTools.render(control.joint_problem, plot_step_joint; title_str="\nConflated Belief")
    push!(plts, plt)

    w, h = plts[1].attr[:size]
    blank_plt = plot(; legend=false, ticks=false, showaxis=false, grid=false, 
        aspectratio=:equal, size=(w, h))

    while length(plts) < num_plots_w
        push!(plts, blank_plt)
    end
    
    for (i, cp) in enumerate(control.indiv_control)
        ai = action(cp)
        plot_step_i = (s=plot_step.s, a=ai, b=cp.belief)
        plt = POMDPTools.render(cp, plot_step_i; title_str="\nAgent $i")
        push!(plts, plt)
    end
    num_plots_h = ceil(Int, length(plts) / num_plots_w)
    adjusted_w = w * num_plots_w 
    adjusted_h = h * num_plots_h * 1.0
    return plot(plts..., layout=(num_plots_h, num_plots_w), size=(adjusted_w, adjusted_h))
end


function POMDPTools.render(control::Conflation, plot_step::NamedTuple)
    num_plots_w = control.joint_problem.num_agents
    plts = []
    
    if !isnothing(get(plot_step, :joint_b, nothing))
        if !isnothing(get(plot_step, :joint_a, nothing))
            plot_step_joint = (s=plot_step.s, a=plot_step.joint_a, b=plot_step.joint_b)
        else
            plot_step_joint = (s=plot_step.s, b=plot_step.joint_b)
        end
        plt_joint = POMDPTools.render(control.indiv_control[1].problem, plot_step_joint; title_str="\nJoint Belief")
        push!(plts, plt_joint)
    end
    
    conflated_b = control.prev_selected_belief
    if !isnothing(get(plot_step, :a, nothing))
        plot_step_conflated = (s=plot_step.s, a=plot_step.a, b=conflated_b)
    else
        plot_step_conflated = (s=plot_step.s, b=conflated_b)
    end
    plt = POMDPTools.render(control.joint_problem, plot_step_conflated; title_str="\nSelected Conflated Belief")
    push!(plts, plt)

    w, h = plts[1].attr[:size]
    blank_plt = plot(; legend=false, ticks=false, showaxis=false, grid=false, 
        aspectratio=:equal, size=(w, h))
    
    while length(plts) < num_plots_w
        push!(plts, blank_plt)
    end
    
    for (i, cp) in enumerate(control.indiv_control)
        ai = action(cp)
        plot_step_i = (s=plot_step.s, a=ai, b=cp.belief)
        plt = POMDPTools.render(cp, plot_step_i; title_str="\nAgent $i")
        push!(plts, plt)
    end
    
    
    # Determine max number of surrogate beliefs for any agent
    num_surrogate_beliefs = zeros(Int, control.joint_problem.num_agents - 1)
    for ii in 2:control.joint_problem.num_agents
        num_surrogate_beliefs[ii-1] = length(control.surrogate_beliefs[ii])
    end
    max_surrogate_beliefs = maximum(num_surrogate_beliefs)
    
    for ii in 1:max_surrogate_beliefs
        for jj in 1:control.joint_problem.num_agents
            if jj == 1
                push!(plts, blank_plt)
            elseif num_surrogate_beliefs[jj-1] >= ii
                eb = control.surrogate_beliefs[jj]
                ai = action(eb.policy, eb.beliefs[ii])
                weight = round(eb.weights[ii], digits=2)
                title_str = "\nAgent $jj, Belief $ii\nWeight: $weight"
                plt = POMDPTools.render(control.joint_problem, (s=plot_step.s, a=ai, b=eb.beliefs[ii]); title_str=title_str)
                push!(plts, plt)
            else
                push!(plts, blank_plt)
            end
        end
    end
    
    
    num_plots_h = ceil(Int, length(plts) / num_plots_w)
    adjusted_w = w * num_plots_w 
    adjusted_h = h * num_plots_h * 1.0
    return plot(plts..., layout=(num_plots_h, num_plots_w), size=(adjusted_w, adjusted_h))
end
