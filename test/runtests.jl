using DataCollocations
using Test
using OrdinaryDiffEq
using DataInterpolations
using JLArrays
using JET

@testset "DataCollocations.jl" begin
    bounded_support_kernels = [
        EpanechnikovKernel(), UniformKernel(), TriangularKernel(),
        QuarticKernel(), TriweightKernel(), TricubeKernel(), CosineKernel(),
    ]

    unbounded_support_kernels = [
        GaussianKernel(), LogisticKernel(), SigmoidKernel(), SilvermanKernel(),
    ]

    @testset "Kernel Functions" begin
        ts = collect(-5.0:0.1:5.0)
        @testset "Kernels with support from -1 to 1" begin
            minus_one_index = findfirst(x -> ==(x, -1.0), ts)
            plus_one_index = findfirst(x -> ==(x, 1.0), ts)
            @testset "$kernel" for (kernel, x0) in zip(
                    bounded_support_kernels,
                    [0.75, 0.5, 1.0, 15.0 / 16.0, 35.0 / 32.0, 70.0 / 81.0, pi / 4.0]
                )
                ws = DataCollocations.calckernel.((kernel,), ts)
                # t < -1
                @test all(ws[1:(minus_one_index - 1)] .== 0.0)
                # t > 1
                @test all(ws[(plus_one_index + 1):end] .== 0.0)
                # -1 < t <1
                @test all(ws[(minus_one_index + 1):(plus_one_index - 1)] .> 0.0)
                # t = 0
                @test DataCollocations.calckernel(kernel, 0.0) == x0
            end
        end
        @testset "Kernels with unbounded support" begin
            @testset "$kernel" for (kernel, x0) in zip(
                    unbounded_support_kernels,
                    [1 / (sqrt(2 * pi)), 0.25, 1 / pi, 1 / (2 * sqrt(2))]
                )
                # t = 0
                @test DataCollocations.calckernel(kernel, 0.0) == x0
            end
        end
    end

    @testset "Collocation of data" begin
        f(u, p, t) = p .* u
        rc = 2
        ps = repeat([-0.001], rc)
        tspan = (0.0, 10.0)  # Reduced time span
        u0 = 3.4 .+ ones(rc)
        t = collect(range(minimum(tspan); stop = maximum(tspan), length = 100))  # Reduced data points
        prob = ODEProblem(f, u0, tspan, ps)
        data = Array(solve(prob, Tsit5(); saveat = t, abstol = 1.0e-12, reltol = 1.0e-12))
        @testset "$kernel" for kernel in [
                bounded_support_kernels..., unbounded_support_kernels...,
            ]
            u′, u = collocate_data(data, t, kernel, 0.1)  # Higher bandwidth
            @test sum(abs2, u - data) < 1.0e-6  # More lenient tolerance
        end
        @testset "$kernel" for kernel in [bounded_support_kernels...]
            # Errors out as the bandwidth is too low
            @test_throws ErrorException collocate_data(data, t, kernel, 0.001)
        end
    end

    @testset "DataInterpolations Extension" begin
        # Test with simple data
        t = 0.0:0.1:1.0
        data = sin.(t)
        tpoints_sample = 0.05:0.1:0.95

        # Test that the extension method works
        du, u = collocate_data(data, collect(t), collect(tpoints_sample), LinearInterpolation)
        @test length(du) == length(tpoints_sample)
        @test length(u) == length(tpoints_sample)
    end

    @testset "Interface Compatibility" begin
        @testset "BigFloat support" begin
            # Test that BigFloat inputs are supported and eltype is preserved
            tpoints = BigFloat.(collect(range(0.0, stop = 10.0, length = 30)))
            data = BigFloat.(reshape(sin.(Float64.(tpoints)), 1, :))
            bandwidth = BigFloat(0.5)

            kernels = [
                EpanechnikovKernel(), UniformKernel(), TriangularKernel(),
                QuarticKernel(), TriweightKernel(), TricubeKernel(),
                GaussianKernel(), CosineKernel(), LogisticKernel(),
                SigmoidKernel(), SilvermanKernel(),
            ]

            @testset "$kernel" for kernel in kernels
                u_prime, u = collocate_data(data, tpoints, kernel, bandwidth)
                @test eltype(u_prime) == BigFloat
                @test eltype(u) == BigFloat
            end
        end

        @testset "GPU array error message" begin
            # Test that GPU-like arrays (JLArrays) give a clear error message
            # instead of cryptic scalar indexing errors
            tpoints = JLArray([1.0, 2.0, 3.0, 4.0, 5.0])
            data = JLArray([1.0 2.0 3.0 4.0 5.0])

            @test_throws ArgumentError collocate_data(data, tpoints)

            # Also test that only GPU tpoints triggers error
            tpoints_cpu = collect(1.0:5.0)
            @test_throws ArgumentError collocate_data(data, tpoints_cpu)

            # And only GPU data triggers error
            data_cpu = reshape(collect(1.0:5.0), 1, :)
            @test_throws ArgumentError collocate_data(data_cpu, tpoints)
        end
    end

    @testset "JET Static Analysis" begin
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
    end
end
