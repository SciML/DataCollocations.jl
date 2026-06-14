using DataCollocations

bounded_support_kernels = [
    EpanechnikovKernel(), UniformKernel(), TriangularKernel(),
    QuarticKernel(), TriweightKernel(), TricubeKernel(), CosineKernel(),
]

unbounded_support_kernels = [
    GaussianKernel(), LogisticKernel(), SigmoidKernel(), SilvermanKernel(),
]
