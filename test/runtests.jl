using Test
using SmartPrior

@testset "SmartPrior" begin
    include("TestGrid.jl")
    include("TestFeatures.jl")
    include("TestPriorNet.jl")
    include("TestLosses.jl")
    include("TestTrain.jl")
    include("TestMetrics.jl")
    include("TestCloncurryIO.jl")
    include("TestSchema.jl")
    include("TestSyntheticFields.jl")
end
