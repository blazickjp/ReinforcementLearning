
module grid
using Gadfly
using Compose
using DataFrames

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
    push!(result, [newState(mdp, state, action), 1, -1])
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
        Theme(alphas = [.5], grid_color="black", minor_label_font_size=24pt,
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
        Theme(alphas = [.5], grid_color="black", major_label_font_size=14pt,
            grid_line_width=2pt, grid_line_style=:solid))
end

# grid = simpleGrid(5, [1, 25], 1)
# grid2 = volcanoGrid(4, [3,7], -100, [4], 10, 0.3, 1)
# plotValueGrid(grid)
# plotValueGrid(grid2)
end