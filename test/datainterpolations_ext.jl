using DataCollocations
using DataInterpolations
using Test

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
