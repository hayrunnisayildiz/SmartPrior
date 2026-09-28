# Tests for the archived grid stack. Run from the repository root:
#   julia --project=. legacy/grid_stack/test/runtests.jl
# The archive needs JLD2 in the active environment (removed from the main
# Project.toml): `julia --project=. -e 'using Pkg; Pkg.add("JLD2")'`.
using Test

include(joinpath(@__DIR__, "..", "GridStack.jl"))
using .GridStack

@testset "GridStack (archived)" begin
    include("TestGrid.jl")
    include("TestFeatures.jl")
    include("TestPriorNet.jl")
    include("TestLosses.jl")
    include("TestTrain.jl")
    include("TestMetrics.jl")
end
