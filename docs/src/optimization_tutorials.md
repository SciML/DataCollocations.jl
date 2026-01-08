# Non-Allocating Forward-Mode L2 Collocation Loss

The following is an example of a loss function over the collocation that
is non-allocating and compatible with forward-mode automatic differentiation.
This approach is useful for high-performance parameter estimation when you
already have the estimated solution and derivatives from `collocate_data`:

```julia
using PreallocationTools

# Assume you have already performed collocation
# estimated_derivative, estimated_solution = collocate_data(data, tpoints, kernel)

# Pre-allocate buffers for forward-mode AD compatibility
du = PreallocationTools.dualcache(similar(prob.u0))

# Create views to avoid allocations during optimization
preview_est_sol = [@view estimated_solution[:, i] for i in 1:size(estimated_solution, 2)]
preview_est_deriv = [@view estimated_derivative[:, i]
                     for i in 1:size(estimated_solution, 2)]

function construct_iip_cost_function(f, du, preview_est_sol, preview_est_deriv, tpoints)
    function (p)
        _du = PreallocationTools.get_tmp(du, p)
        vecdu = vec(_du)
        cost = zero(first(p))
        for i in 1:length(preview_est_sol)
            est_sol = preview_est_sol[i]
            f(_du, est_sol, p, tpoints[i])
            vecdu .= vec(preview_est_deriv[i]) .- vec(_du)
            cost += sum(abs2, vecdu)
        end
        sqrt(cost)
    end
end

# Create the cost function
cost_function = construct_iip_cost_function(
    f, du, preview_est_sol, preview_est_deriv, tpoints)

# Use with your optimization framework
# optf = Optimization.OptimizationFunction((x, p) -> cost_function(x), adtype)
# optprob = Optimization.OptimizationProblem(optf, initial_params)
# result = Optimization.solve(optprob, optimizer)
```

This cost function:
- Uses `PreallocationTools.dualcache` for dual number compatibility
- Pre-computes views to avoid repeated memory allocations
- Computes the L2 norm of the difference between `f(u,p,t)` and the estimated derivatives
- Returns a scalar cost suitable for gradient-based optimization