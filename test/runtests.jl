using Test
using SmartPrior

@testset "SmartPrior" begin
    include("TestSchema.jl")
    include("TestDesurvey.jl")
    include("TestKeivitsaIO.jl")
    include("TestSyntheticFields.jl")
end
