using DataCollocations
using Test
include(joinpath(@__DIR__, "shared", "kernels_setup.jl"))

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
