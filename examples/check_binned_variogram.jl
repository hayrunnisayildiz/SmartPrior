# Checks that `binned_variogram` (streaming) gives exactly the same bins as
# `collect_pairs` + `bin_experimental` (stored pairs). No GTK data needed.
#
# Run:  julia --project=. examples/check_binned_variogram.jl

ENV["SMARTPRIOR_WORK"] = mktempdir()
include(joinpath(@__DIR__, "feasibility_loho.jl"))
using Random

function check_once(seed, n)
    rng = Xoshiro(seed)
    nholes = max(2, n ÷ 40)
    hole = [string("H", rand(rng, 1:nholes)) for _ in 1:n]
    hole[rand(rng, 1:n)] = ""                      # a merged point with no hole id
    x = 1000 .* rand(rng, n)
    y = 1000 .* rand(rng, n)
    z = -400 .* rand(rng, n)
    # One pair exactly on a lag edge (bins are (lo, hi], so it belongs to the lower bin).
    edge = collect(range(0.0, 500.0; length = N_LAGS + 1))[3]
    x[1] = 0.0; x[2] = edge; y[2] = y[1]; z[2] = z[1]
    v = randn(rng, n)
    maxlag_h, maxlag_v, dip = 500.0, 200.0, HORIZONTAL_DIP_DEG

    dh, gh, dv, gv = collect_pairs(x, y, z, v, hole;
                                   maxlag_h = maxlag_h, maxlag_v = maxlag_v, dip_deg = dip)
    bh, eh = bin_experimental(dh, gh, N_LAGS, maxlag_h)
    bv, ev = bin_experimental(dv, gv, N_LAGS, maxlag_v)

    bh2, eh2, bv2, ev2, nh, nv = binned_variogram(x, y, z, v, hole;
        maxlag_h = maxlag_h, maxlag_v = maxlag_v, dip_deg = dip, n_lags = N_LAGS)

    ok = bh == bh2 && eh == eh2 && bv == bv2 && ev == ev2 &&
         nh == length(dh) && nv == length(dv)
    println("seed $seed n $n: ", ok ? "identical" : "DIFFERENT",
            " (pairs h $(length(dh)), v $(length(dv)))")
    return ok
end

all_ok = all(check_once(s, n) for (s, n) in ((1, 300), (2, 1000), (3, 2500)))
println(all_ok ? "OK: streaming bins are bit-identical" : "FAIL")
all_ok || exit(1)
