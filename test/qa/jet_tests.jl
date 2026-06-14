using DataCollocations
using Test
using JET

# Test kernel functions for type stability
@testset "calckernel optimization" begin
    kernels = [
        EpanechnikovKernel(), UniformKernel(), TriangularKernel(),
        QuarticKernel(), TriweightKernel(), TricubeKernel(),
        GaussianKernel(), CosineKernel(), LogisticKernel(),
        SigmoidKernel(), SilvermanKernel(),
    ]
    @testset "$kernel" for kernel in kernels
        JET.@test_opt target_modules = (DataCollocations,) DataCollocations.calckernel(kernel, 0.5)
    end
end

# Test main collocate_data function
@testset "collocate_data optimization" begin
    data = rand(2, 20)
    tpoints = collect(range(0.0, 1.0, length = 20))
    JET.@test_opt target_modules = (DataCollocations,) collocate_data(data, tpoints, TriangularKernel(), 0.1)
end
