module DataCollocations

using LinearAlgebra
using ArrayInterface: fast_scalar_indexing

export collocate_data
export EpanechnikovKernel, UniformKernel, TriangularKernel, QuarticKernel
export TriweightKernel, TricubeKernel, GaussianKernel, CosineKernel
export LogisticKernel, SigmoidKernel, SilvermanKernel

include("collocation.jl")

end
