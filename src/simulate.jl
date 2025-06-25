using POMDPs
using POMDPTools
import SARSOP
using Random
using Plots
using StatsBase
using JLD2
using Printf
using LinearAlgebra
using DataFrames
using ProgressMeter
using MultiAgentPOMDPProblems
    
include("suggested_action_policies.jl")

mutable struct SimulateResults
    step_count::Int
    cum_reward::Float64
    cum_discounted_rew::Float64
    num_beliefs::Dict{Int, Vector{Int}}
end

function Base.show(io::IO, results::SimulateResults)
    println(io, "SimulateResults")
    println(io, "\tStep Count                   : $(results.step_count)")
    println(io, "\tCumulative Reward            : $(results.cum_reward)")
    println(io, "\tCumulative Discounted Reward : $(results.cum_discounted_rew)")
    ks = keys(results.num_beliefs)
    tot = 0
    for k in sort(collect(ks))
        println(io, "\tMax Num Beliefs $k            : $(maximum(results.num_beliefs[k]))")
        println(io, "\tAverage Num Beliefs $k        : $(mean(results.num_beliefs[k]))")
        println(io, "\tStd Num Beliefs $k            : $(std(results.num_beliefs[k]))")
        tot += sum(results.num_beliefs[k])
    end
end

function run_simulation(
    problem::POMDP{S, A, O}, # The problem used to drive the simulation
    init_state::S,
    control::MultiAgentControlStrategy;
    max_steps::Int=35,
    seed::Int=42,
    show_plots::Bool=false,
    text_output::Bool=false,
    joint_control=nothing
) where {S, A, O}
    rng = MersenneTwister(seed)
    
    num_agents = problem.num_agents
    γ = discount(problem)
    
    # Initialize the results struct
    num_beliefs_dict = Dict{Int, Vector{Int}}()
    for jj in 2:num_agents
        num_beliefs_dict[jj] = Vector{Int}()
    end
    results = SimulateResults(0, 0.0, 0.0, num_beliefs_dict)
    
    s = deepcopy(init_state)
    
    stop_simulating = false
    while !stop_simulating
        results.step_count += 1
        
        num_beliefs_before = Dict{Int, Int}()
        num_beliefs_after = Dict{Int, Int}()
        
        if text_output
            for jj in 2:num_agents
                num_beliefs_before[jj] = length(control.surrogate_beliefs[jj])
            end
        end
        
        # Get actions based on the control strategy
        act, info = action_info(control)
        
        if text_output
            for jj in 2:num_agents
                num_beliefs_after[jj] = length(control.surrogate_beliefs[jj])
            end
        end
        
        # Number of beliefs after shared action pruning
        if control isa Conflation
            for jj in 2:num_agents
                push!(results.num_beliefs[jj], length(control.surrogate_beliefs[jj]))
            end
        else
            for jj in 2:num_agents
                push!(results.num_beliefs[jj], 1)
            end
        end
        
        if !isnothing(joint_control)
            joint_a, _ = action_info(joint_control)
        end
        
        # Generate the next state, observation, and reward (simulate the problem)
        (sp, o, r) = @gen(:sp, :o, :r)(problem, s, act, rng)
        
        # Any text info output here for simulation inspection/debugging/etc.
        if text_output
            println("Step                       : $(results.step_count)")
            println("State                      : ", s)
            println("Actions                    : ", act)
            println("Next State                 : ", sp)
            println("Observations               : ", o)
            println("Reward                     : ", r)
            println("Nuber of surrgoate beliefs")
            for jj in 2:num_agents
                println("\tAgent $jj            : $(num_beliefs_before[jj]) -> $(num_beliefs_after[jj])")
            end
            println()
        end
        
        # Update the results struct
        results.cum_reward += r
        results.cum_discounted_rew += r * γ^(results.step_count-1)
        
        # Plotting options to visualize the simulation and policy decisions
        if show_plots
            if !isnothing(joint_control)
                if joint_control isa ConflateJoint
                    joint_conflated_b = conflate_beliefs(joint_control)
                    plot_step = (s=s, a=act, joint_a=joint_a, joint_b=joint_conflated_b)
                elseif joint_control isa SinglePolicy || joint_control isa JointPolicy
                    plot_step = (s=s, a=act, joint_a=joint_a, joint_b=joint_control.belief)
                else
                    throw(ArgumentError("Invalid joint control for simulation plotting: $(typeof(joint_control))"))
                end
            else
                plot_step = (s=s, a=act)
            end
            plt = POMDPTools.render(control, plot_step)
            display(plt)
        end
        
        # Update the beliefs. This is different than the normal POMDPs.jl process because
        # we need to update the beliefs differently based on the control strategy.
        update_belief!(control, act, o)
        
        if !isnothing(joint_control)
            update_belief!(joint_control, act, o)
        end
            
        # Update the state and check if the simulation should continue
        s = sp
        
        if isterminal(problem, s) || results.step_count >= max_steps
            stop_simulating = true
        end
    end
    
    return results
end
