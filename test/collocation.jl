using DataCollocations
using OrdinaryDiffEq
using Test
include(joinpath(@__DIR__, "shared", "kernels_setup.jl"))

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
