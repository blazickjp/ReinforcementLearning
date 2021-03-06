
using Gadfly
using Compose
using DataFrames
using StatsBase

abstract type Gridworld end 

mutable struct simpleGrid <: Gridworld
    width::Number
    endstates::Any
    discount::Number
end

mutable struct volcanoGrid <: Gridworld
    width::Number
    penaltyStates::Any
    penalty::Number
    rewardStates::Any
    reward::Number
    probSlip::Number
    discount::Number
end

"Check if current state is in End State"
function isEnd(mdp::simpleGrid, state)
    return state in mdp.endstates
end

"Check if current state is in End State"
function isEnd(mdp::volcanoGrid, state)
    if state ∈ mdp.penaltyStates
        return true
    elseif state ∈ mdp.rewardStates
        return true
    else
        return false
    end
end

"Retunrs all possible actions given current state"
function actions(mdp::Gridworld, state)
    actions = ["left", "right", "up", "down"]
    if isEnd(mdp, state)
        return []
    end
    if state in 1:mdp.width
        filter!(x->x != "up", actions)
    end
    if state in mdp.width^2-mdp.width + 1:mdp.width^2
        filter!(x->x != "down", actions)
    end
    if mod(state, mdp.width) == 0
        filter!(x->x != "right", actions)
    end
    if mod(state, mdp.width) == 1
        filter!(x->x !="left", actions)
    end
    return actions
end

function newState(mdp::Gridworld, state::Number, action::String)
    if action == "right"
        return state + 1
    elseif action == "left"
        return state - 1
    elseif action == "down"
        return state + mdp.width
    elseif action == "up"
        return state - mdp.width
    end
end

"Helper function to return all states"
function states(mdp::Gridworld)
    return 1:mdp.width^2
end

"Returns tuple of (New State, Probability, Reward)"
function succProbReward(mdp::simpleGrid, state, action)
    result = []
    newS = newState(mdp, state, action)
    reward = isEnd(mdp, newS) ? 0 : -1
    push!(result, [newS, 1, reward])
    return result
end

"Returns tuple of (New State, Probability, Reward)"
function succProbReward(mdp::volcanoGrid, state, action)
    result = []
    n = length(actions(mdp, state)) - 1
    for i in actions(mdp, state)
        if newState(mdp, state, i) ∈ mdp.penaltyStates
            reward = mdp.penalty
        elseif newState(mdp, state, i) ∈ mdp.rewardStates
            reward = mdp.reward
        else
            reward = -1
        end
        if i == action
            push!(result, [newState(mdp, state, i), 1 - mdp.probSlip, reward])
        else
            push!(result, [newState(mdp, state, i), mdp.probSlip / n, reward])
        end
    end
    return result
end

"Sample from actions according to probabilities and return new state, reward"
function sampleAction(mdp::Gridworld, state, action)
    newStates = []
    probs = []
    rewards = []
    for (newState, prob, reward) in succProbReward(mdp, state, action)
        push!(newStates, newState)
        push!(probs, prob)
        push!(rewards, reward)
    end 
    idx = sample(1:length(probs), ProbabilityWeights(convert(Array{Float64,1}, probs)))
    return newStates[idx], rewards[idx] 
end

function TDIteraation(mdp::Gridworld, episodes, exploration_prob=0.2, startState=1, α=.001)
    value =  zeros(length(states(mdp)))
    delta = []
    function Q(mdp::Gridworld, state, action)
        q = sum(
                prob * (reward + mdp.discount * value[Int(newState)]) 
                for (newState, prob, reward) in succProbReward(mdp, state, action)
            )
        return q
    end
    function getAction(mdp::Gridworld, state)
        if rand() < exploration_prob
            return rand(actions(mdp, state))
        else
            return maximum((Q(mdp, state, action), action) for action in actions(mdp, state))[2]
        end
    end
    # For each trial
    for i ∈ 1:episodes
        state = startState
        # Run each episode
        while true
            action = getAction(mdp, state)
            newState, reward = sampleAction(mdp, state, action)
            before = value[Int(state)]
            value[Int(state)] += α .* (reward .+ mdp.discount .* value[Int(newState)] .- value[Int(state)])
            push!(delta, before - value[Int(state)])
            if isEnd(mdp, newState) break end
            state = newState
        end
    end
    #plot(1:length(delta), delta, Geom.line)
    return DataFrame(value=value, state = states(mdp))
end

"Impliments Value iteration on simpleGrid Struct"
function valueIteration(mdp::Gridworld)
    # Initialize value and policy arrays
    value =  Array{Float64}(undef, length(states(mdp)))
    p = Array{String}(undef, length(states(mdp)))
    
    for (i, state) in enumerate(states(mdp))
        value[i] = 0
    end
    # Q Function
    function Q(mdp::Gridworld, state, action)
        q = sum(
                prob * (reward + mdp.discount * value[Int(newState)]) 
                for (newState, prob, reward) in succProbReward(mdp, state, action)
            )
        return q
    end
    # This should be until convergence
    for j in 1:100
        vNew =  Array{Float64}(undef, length(states(mdp)))
        for (i, state) in enumerate(states(mdp))
            if isEnd(mdp, state)
                vNew[i] = 0
            else 
                vNew[i] = maximum([Q(mdp, state, action) for action in actions(mdp, state)])
            end
        end
        value = vNew

        # Get policy
        for (i, state) in enumerate(states(mdp))
            if isEnd(mdp, state)
                p[i] = "None"
            else
                p[i] = maximum((Q(mdp, state, action), action) for action in actions(mdp, state))[2]
            end
        end
    end
    out = []
    for (i, state) in enumerate(states(mdp))
        push!(out, (state, value[i], p[i]))
    end
    return out
end

"Generates plot with V*(s) values for each state in Grid"
function plotValueGrid(mdp::simpleGrid)
    result = valueIteration(mdp)
    df = DataFrame(
        [(y=y, x=x,)
            for y in -(1:mdp.width) 
            for x in 1:mdp.width
        ])
    df[!, :label] = [string(result[2]) for result in result]
    df[!, :endState] = [isEnd(mdp, state) for state in states(mdp)]
    end_coords = filter(row -> row.endState == true, df)
    x = mdp.width + .5 
    plot(df, x=:x, y=:y, label=:label, Geom.label,
        Guide.xticks(ticks=0.5:x, label=false),
        Guide.yticks(ticks=-(0.5:x), label=false), 
        Guide.title("Simple Gridworld V(s) ∀ s ∈ S"),
        Guide.xlabel(nothing), Guide.ylabel(nothing),
        Guide.Annotation(compose(context(),             
        (context(), 
            rectangle(
                end_coords[!,:x] .- .5,
                end_coords[!,:y] .- .5,
                ones(size(end_coords)[1]),
                ones(size(end_coords)[1])), 
            fill("green"),fillopacity(0.1), stroke("green")),
        )),
        Theme(alphas = [.5], grid_color="black",
            grid_line_width=2pt, grid_line_style=:solid))
end

function plotValueGrid(mdp::volcanoGrid)
    result = valueIteration(mdp)
    df = DataFrame(
        [(y=y, x=x,)
            for y in -(1:mdp.width) 
            for x in 1:mdp.width
        ])
    df[!, :label] = [string(round(result[2], digits=2)) for result in result]
    df[!, :rewardStates] = [state ∈ mdp.rewardStates for state in states(mdp)]
    df[!, :penaltyStates] = [state ∈ mdp.penaltyStates for state in states(mdp)]
    reward_coords = filter(row -> row.rewardStates == true, df)
    penalty_coords = filter(row -> row.penaltyStates == true, df)
    x = mdp.width + .5 
    plot(df, x=:x, y=:y, label=:label, Geom.label,
        Guide.xticks(ticks=0.5:x, label=false),
        Guide.yticks(ticks=-(0.5:x), label=false), 
        Guide.title("Volcano Gridworld V(s) ∀ s ∈ S"),
        Guide.xlabel(nothing), Guide.ylabel(nothing),
        Guide.Annotation(compose(context(),             
        (context(), 
            rectangle(
                reward_coords[!,:x] .- .5,
                reward_coords[!,:y] .- .5,
                ones(size(reward_coords)[1]),
                ones(size(reward_coords)[1])), 
            fill("green"),fillopacity(0.1), stroke("green")),
        (context(), 
            rectangle(
                penalty_coords[!,:x] .- .5,
                penalty_coords[!,:y] .- .5,
                ones(size(penalty_coords)[1]),
                ones(size(penalty_coords)[1])), 
            fill("red"),fillopacity(0.1), stroke("red")),
        )),
        Theme(alphas = [.5], grid_color="black",
            grid_line_width=2pt, grid_line_style=:solid))
end

# grid = simpleGrid(5, [1, 25], 1)
# grid2 = volcanoGrid(4, [3,7], -100, [4], 10, 0.3, 1)
# plotValueGrid(grid)
# plotValueGrid(grid2)


