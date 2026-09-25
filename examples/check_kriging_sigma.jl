# Checks that `kriging_predict` returns a standard deviation, not a variance.
# Known case: two uncorrelated points (spherical, range 1 m, sill 100, no nugget),
# query 1 km away. Ordinary kriging mean = 0.5, variance = 100 + 100/2 = 150,
# so the standard deviation must be √150 ≈ 12.247. No GTK data needed.
#
# Run:  julia --project=. examples/check_kriging_sigma.jl

ENV["SMARTPRIOR_WORK"] = mktempdir()
include(joinpath(@__DIR__, "feasibility_loho.jl"))

fit = VarioFit("spherical", 0.0, 100.0, 100.0, 1.0, 1.0, 0.0, 0, 0, 0, 0, false)
μ, σ = kriging_predict([0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0],
                       [1000.0], [0.0], [0.0], fit)

println("GeoStats returns variance as σ: ", kriging_sigma_is_variance())
println("μ = ", μ[1], "  (expected 0.5)")
println("σ = ", σ[1], "  (expected ", sqrt(150.0), ")")
ok = isapprox(μ[1], 0.5; atol = 1.0e-8) && isapprox(σ[1], sqrt(150.0); rtol = 1.0e-6)
println(ok ? "OK: kriging σ is a standard deviation" : "FAIL")
ok || exit(1)
