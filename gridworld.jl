using Gadfly
using Compose
using DataFrames

struct Gridworld
    width::Number
    endstates::Any
    discount::Number
end

"Check if current state is in End State"
function isEnd(mdp::Gridworld, state)
    return state in mdp.endstates
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

"Helper function to return all states"
function states(mdp::Gridworld)
    return 1:mdp.width^2
end

"Returns tuple of (New State, Probability, Reward)"
function succProbReward(mdp::Gridworld, state, action)
    result = []
    if action == "right"
        push!(result, [state + 1, 1, -1])
    elseif action == "left"
        push!(result, [state - 1, 1, -1])
    elseif action == "down"
        push!(result, [state + mdp.width, 1, -1])
    elseif action == "up"
        push!(result, [state - mdp.width, 1, -1])
    end
    return result
end

"Impliments Value iteration on Gridworld Struct"
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
function plotValueGrid(mdp::Gridworld)
    result = valueIteration(grid)
    print(result)
    df = DataFrame(
        [(y=y, x=x,)
            for y in -(1:grid.width) 
            for x in 1:grid.width
        ])
    df[!, :label] = [string(result[2]) for result in result]
    df[!, :endState] = [isEnd(mdp, state) for state in states(mdp)]
    end_coords = filter(row -> row.endState == true, df)
    x = grid.width + .5 
    plot(df, x=:x, y=:y, label=:label, Geom.label,
        Guide.xticks(ticks=0.5:x, label=false),
        Guide.yticks(ticks=-(0.5:x), label=false), 
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

grid = Gridworld(5, [1, 25], 1)
plotValueGrid(grid)


