# Isolated env for GeoStats.jl ordinary-kriging baseline.
#
# Cannot be merged into the main SmartPriorMT Project.toml: GeoStats needs
# Meshes ≥0.57, which conflicts with MTGeophysics → CairoMakie → Makie 0.23.
#
# Install / update:
#   julia --project=examples/kriging_env -e 'using Pkg; Pkg.instantiate()'
#
# Invoked by examples/kriging_cloncurry_petro.jl — do not add SmartPriorMT here.
