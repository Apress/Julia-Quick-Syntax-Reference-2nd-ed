# Shelling Segregation model

## Instructions 

# Source: Thomas C. Schelling (1971) Dynamic models of segregation, The Journal of Mathematical Sociology, 1:2, 143-186, DOI: [10.1080/0022250X.1971.9989794](https://doi.org/10.1080/0022250X.1971.9989794) 

# In this exercise you will implement the [Shelling Segregation Model](https://en.wikipedia.org/wiki/Schelling%27s_model_of_segregation), a classic agent-based model in the social sciences, which introduces the concepts of emergent macro-behaviours and tipping points, and which led to its author (Thomas Schelling) being awarded the Nobel Prize in Economics in 2005.

# It was one of the very first results obtained from running simulations, especially in the social sciences.
# The world is modelled as a gridded space inhabited by two groups (at the time, racial and segregation issues concerning the co-existence of "blacks" and "whites" were topical).
# Each group has a preference to live in a neighbourhood inhabited by agents of its own type, with a certain "tolerance". If the proportion of agents of its own type in the neighbourhood falls below this "tolerance" level, the agent will try to move to a place with a higher proportion of agents of its own type.
# The main result of the paper was that even a relatively mild preference for the presence of similar agents would have led to a segregated world, and that there was a specific threshold (between 30% and 40%) that drove these two completely different outcomes, i.e. a "tipping point".

# The simulation algorithm then works like this:
# At each step, it looks at each agent, checks whether it is "happy" with its current location (looking at the proportion of its own types in the neighbourhood), and if not, moves the agent to a position where it would be (setting its previous location as empty).

# There are several "generality" vs. "specificity" ways to code the above algorithm. On the one hand you could hard code the two agent types, e.g. as `blue` and `red`, on the other hand you could be very generic and create an abstract type `agent` and a concrete class for each agent type.

# The skeleton below suggests an intermediate approach with only one `Agent` class and the type of agent encoded as an integer, where 0 represents an empty cell. Feel free to use it or develop your own algorithm from scratch!

# Source: Thomas C. Schelling (1971) Dynamic models of segregation, The Journal of Mathematical Sociology, 1:2, 143-186, DOI: [10.1080/0022250X.1971.9989794](https://doi.org/10.1080/0022250X.1971.9989794) 


# ------------------------------------------------------------------------------
# ## 1) Setting the environment...

cd(@__DIR__)         
using Pkg             
Pkg.activate(".")   
# If using a Julia version different than 1.10 or 1.11 please uncomment and run the following line (reproductibility guarantee will hower be lost). If your version of Julia can't find a suitable combination, try running `Pkg.up();Pkg.resolve()`
# Pkg.resolve()   
Pkg.instantiate()
using Random
Random.seed!(123)
using Plots

# ------------------------------------------------------------------------------
# ## 2) Defining the `Agent` and `Env` classes... 
mutable struct ShellingAgent
    gid::Int64
end
mutable struct Env
    nR::Int64                       # number of rows
    nC::Int64                       # number of columns
    similarityThreeshold::Float64   # threeshold for agents to be "happy" with their location
    neighborhood::Int64             # how far looking for "similar" agents
    nSteps::Int64                   # number of iteractive steps to employ
    cells::Vector{ShellingAgent}            # total cells in the environment
    gids::Vector{Int64}             # ids of the agents types (or "groups")
    grsizes::Vector{Int64}          # number of agents per group
end

# ------------------------------------------------------------------------------
# ## 3) Defining some utility functions... 
xyToId(x,y,nR,nC)  =  nR*(x-1)+y # Alternative approach: use CartesianIndexes
iDToXY(id,nR,nC)   =  Int(floor((id-1)/nR)+1), (id-1)%(nR)+1
printableGrid(env) = reshape([a.gid for a in env.cells],env.nR,env.nC)

# ------------------------------------------------------------------------------
# ## 4) Defining the main functions of the algorithm...
"""
   getNeighbours(x,y,env,gid=nothing)

Return the number of total neighbours if `gid` is `nothing` (skipping the empty cells) or of a specific `gid` if this one is provided.

"""
function getNeighbours(x,y,env,gid=nothing)
    board  = reshape(env.cells,env.nR,env.nC)
    region = board[max(1,y-env.neighborhood):min(nR,y+env.neighborhood),max(1,x-env.neighborhood):min(nC,x+env.neighborhood)]
    ## [...] Write your code here
end

"""
   isHappy(x,y,a,env)

Return whether the specific agent `a` is happy at his current location
"""
function isHappy(x,y,a,env)
   ## [...] Write your code here
end


"""
   reallocatePoints!(env)

Loop over all the cells and if an agent on that location is unhappy, it moves it to a location where it is happy and set the departing cell as empty (i.e. occupied by an agent whose gid is zero).
It returns the share of agents that were happy before the move.
"""
function reallocatePoints!(env)
    happyCount = 0
    for (i,a) in enumerate(env.cells)
        ## [...] Write your code here
    end
    return happyCount/sum(env.grsizes)
end

"""
    modelrun!(env)

Run the reallocation algorithm for the given steps printing a heatmap at each iteration.
Also, print at the end the chart of the happy agents by epoch
"""
function modelrun!(env)
    outplot = heatmap(printableGrid(env), legend=nothing, title="START", color=mypal,aspect_ratio=env.nR/env.nC, size=(600,600*env.nR/env.nC))
    nHappyCount = Float64[]
    display(outplot)
    for i in 1:env.nSteps
        println("Running iteration $i...")
        nHappy = reallocatePoints!(env)
        push!(nHappyCount,nHappy)
        outplot = heatmap(printableGrid(env), legend=nothing, title="Iteration $i", color=mypal,aspect_ratio=env.nR/env.nC, size=(600,600*env.nR/env.nC))
        display(outplot)
    end
    happyCountPlot = plot(nHappyCount,title="Share of happy agents by iteration")
    display(happyCountPlot)
end

# ------------------------------------------------------------------------------
# ## 5) Setting the parameters of the specific simulation to run...

# Parameters...
nR         = 200
nC         = 200
nSteps     = 20
similarityThreeshold = 0.4         # Agent is happy if at least 40% similar
neighborhood = 5                   # Defining how far looking for similar agents
mypal        = [:white,:red,:blue] # First colour is for the empty cell
gids         = [1,2]               # Gid 0 is reserved for empty cell
grShares     = [0.4,0.4]           # Shares of cells occupied by agents, by type


# ------------------------------------------------------------------------------
# ## 6) Initialising the simulation with the given parameters...
nCells = nR*nC
nGroups = length(gids)
grsizes= Int.(ceil.(nCells .* grShares))

cells  = fill(ShellingAgent(0),nCells)
count = 1
for g in 1:nGroups
    [cells[j] = ShellingAgent(gids[g]) for j in count:count+grsizes[g]-1]
    count += grsizes[g]
end 

shuffle!(cells)
env = Env(nR,nC,similarityThreeshold,neighborhood,nSteps,cells,gids,grsizes)
heatmap(printableGrid(env), legend=nothing, title="Iteration 0", color=mypal,aspect_ratio=env.nR/env.nC, size=(600,600*env.nR/env.nC))

# ------------------------------------------------------------------------------
# ## 7) Running the model...
modelrun!(env)

# ------------------------------------------------------------------------------
# ## 8) Optional variations...

# You can try to implement variations of this model. Some ideas:

# - implement multiple agent groups;
# - run Monte-Carlo simulations with respect to the tolerance threshold, with it being a property of the agent rather than of the group, perhaps with the group having a certain distribution of it and each agent of that group sampling from it;
# - implement two thresholds, with one to define happiness and one to define the actual decision to move (to consider costs associated with relocation);
# - ....