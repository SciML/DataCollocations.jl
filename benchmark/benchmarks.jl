using DataCollocations, BenchmarkTools
using StableRNGs

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

n = 500
t = collect(range(0.0, 10.0, length = n))
data = sin.(t) .+ 0.05 .* randn(rng, n)
data2 = hcat(sin.(t), cos.(t))'  # 2-state data

# =============================================================================
# collocate_data — derivative estimation via kernel collocation
# =============================================================================

SUITE["collocate"] = BenchmarkGroup()

SUITE["collocate"]["epanechnikov"] = @benchmarkable collocate_data(
    $data, $t, EpanechnikovKernel(), 0.3
)
SUITE["collocate"]["gaussian"] = @benchmarkable collocate_data(
    $data, $t, GaussianKernel(), 0.3
)
SUITE["collocate"]["triangular"] = @benchmarkable collocate_data(
    $data, $t, TriangularKernel(), 0.3
)
SUITE["collocate"]["logistic"] = @benchmarkable collocate_data(
    $data, $t, LogisticKernel(), 0.3
)
SUITE["collocate"]["multistate"] = @benchmarkable collocate_data(
    $data2, $t, EpanechnikovKernel(), 0.3
)
