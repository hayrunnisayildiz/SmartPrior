using Test
using SmartPriorMT

@testset "SmartPriorMT" begin
    include("TestGrid.jl")
    include("TestGravity.jl")
    include("TestMT1D.jl")
    include("TestFeatures.jl")
    include("TestPriorNet.jl")
    include("TestLosses.jl")
    include("TestTrain.jl")
    include("TestExport.jl")
    include("TestSynthetic.jl")
    include("TestMetrics.jl")
    include("TestProfile2D.jl")
    include("TestRealDataIO.jl")
    include("TestKeivitsaIO.jl")
    include("TestCloncurryIO.jl")
end
