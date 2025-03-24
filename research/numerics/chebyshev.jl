using Plots

struct ChebyshevState
    coeffs::Vector{BigFloat}
    xMin::BigFloat
    xMax::BigFloat
    matchLeft::Bool
    matchRight::Bool
end

function ChebyshevState(f::Function, xMin::BigFloat, xMax::BigFloat, steps::Int, matchLeft::Bool=false, matchRight::Bool=false) :: ChebyshevState
    coeffs = Vector{BigFloat}(undef, steps)
    fill!(coeffs, BigFloat(0))
    for j in 1:steps
        for k in 1:steps
            xRel = BigFloat(0.5) * (BigFloat(1) + cos(BigFloat(π) * (BigFloat(k) - BigFloat(0.5)) / BigFloat(steps)))
            x = xMin + (xMax - xMin) * xRel
            fVal = BigFloat(f(x))
            weight = cos(BigFloat(π) * (BigFloat(j - 1)) * (BigFloat(k) - BigFloat(0.5)) / BigFloat(steps))
            coeffs[j] += BigFloat(2) * fVal * weight / BigFloat(steps)
        end
    end
    evaluate_local(x::BigFloat) = begin
        scale = BigFloat(4) / (xMax - xMin)
        xr2 = -BigFloat(2) + (x - xMin) * scale
        d = BigFloat(0)
        dd = BigFloat(0)
        for j in steps:-1:2
            temp = d
            d = xr2*d - dd + coeffs[j]
            dd = temp
        end
        BigFloat(0.5)*xr2*d - dd + BigFloat(0.5) * coeffs[1]
    end
    if steps > 1
        xMinOffs = matchLeft ? BigFloat(f(xMin)) - evaluate_local(xMin) : BigFloat(0)
        xMaxOffs = matchRight ? BigFloat(f(xMax)) - evaluate_local(xMax) : BigFloat(0)
        a = BigFloat(0.5)*(xMaxOffs + xMinOffs)
        b = BigFloat(0.5)*(xMaxOffs - xMinOffs)
        coeffs[1] += BigFloat(2) * a
        if steps >= 2
            coeffs[2] += b
        end
    end
    ChebyshevState(coeffs, xMin, xMax, matchLeft, matchRight)
end

function chebyshevNodes(expansion::ChebyshevState) :: Vector{BigFloat}
    n = length(expansion.coeffs)
    [expansion.xMin + (expansion.xMax - expansion.xMin) * BigFloat(0.5) * (BigFloat(1) + cos(BigFloat(π) * (BigFloat(i) - BigFloat(0.5)) / BigFloat(n))) for i in 1:n]
end

function evaluate(expansion::ChebyshevState, x::Real) :: BigFloat
    x_big = BigFloat(x)
    scale = BigFloat(4) / (expansion.xMax - expansion.xMin)
    xr2 = -BigFloat(2) + (x_big - expansion.xMin)*scale
    d = BigFloat(0)
    dd = BigFloat(0)
    n = length(expansion.coeffs)
    for j in n:-1:2
        temp = d
        d = xr2*d - dd + expansion.coeffs[j]
        dd = temp
    end
    BigFloat(0.5)*xr2*d - dd + BigFloat(0.5)*expansion.coeffs[1]
end

f(x) = exp(x)

xMin = BigFloat(-2.0)
xMax = BigFloat(2.0)
steps = 8
println("Approximating function within domain [$(xMin), $(xMax)] in $(steps) steps...")
approx = ChebyshevState(f, xMin, xMax, steps, false, false)
x_vals_big = collect(range(xMin, xMax, length=500))
true_vals_big = [BigFloat(f(x)) for x in x_vals_big]
approx_vals_big = [evaluate(approx, x) for x in x_vals_big]
error_vals_big = [abs(t - a) for (t, a) in zip(true_vals_big, approx_vals_big)]
x_vals = Float64.(x_vals_big)
true_vals = Float64.(true_vals_big)
approx_vals = Float64.(approx_vals_big)
error_vals = Float64.(error_vals_big)
p1 = plot(x_vals, true_vals, label="f(x)", lw=2, legend=:topright, title="Chebyshev Approximation", xlabel="x", ylabel="y")
plot!(p1, x_vals, approx_vals, label="$(steps)-term approx", lw=2, ls=:dash)
nodes = chebyshevNodes(approx)
node_vals = [evaluate(approx, x) for x in nodes]
scatter!(p1, Float64.(nodes), Float64.(node_vals), label="Chebyshev", marker=:circle, color=:red)
p2 = plot(x_vals, error_vals, label="Error", lw=2, title="Error Plot", xlabel="x", ylabel="|f(x) - a(x)|")
p_combined = plot(p1, p2, layout=(2, 1))
display(p_combined)

println("Full Precision Coeffs")
for c in approx.coeffs
    println("\t$(c)")
end

println("f64 Precision Coeffs")
for c in approx.coeffs
    println("\t$(Float64.(c))")
end

readline()
