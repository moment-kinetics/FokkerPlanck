"""
"""
module calculus

export integral

using ..type_definitions: mk_float, mk_int
using ..coordinates: finite_element_coordinate
"""
Computes the integral of the integrand, using the input wgts
"""
function integral(integrand::AbstractArray{mk_float,1},
                    wgts::Array{mk_float,1})
    # n is the number of grid points
    n = length(wgts)
    # initialize 'integral' to zero before sum
    integral = 0.0
    @boundscheck n == length(integrand) || throw(BoundsError(integrand))
    @inbounds for i ∈ 1:n
        integral += integrand[i]*wgts[i]
    end
    return integral
end

"""
Compute the 2D integral `∫d^2vperp.dvpa prefactor(vperp,vpa)*integrand`

In this variant `vperp` and `vpa` should be `coordinate` objects.
"""
function integral(prefactor::Function,
                integrand::AbstractArray{mk_float,2},
                vperp::finite_element_coordinate,
                vpa::finite_element_coordinate)
    @boundscheck (vpa.n, vperp.n) == size(integrand) || throw(BoundsError(integrand))
    vperp_grid = vperp.grid
    vperp_wgts = vperp.wgts
    vpa_grid = vpa.grid
    vpa_wgts = vpa.wgts
    integral = 0.0
    for ivperp ∈ eachindex(vperp_grid), ivpa ∈ eachindex(vpa_grid)
        integral += prefactor(vperp_grid[ivperp], vpa_grid[ivpa]) *
                    integrand[ivpa, ivperp] * vperp_wgts[ivperp] * vpa_wgts[ivpa]
    end
    return integral
end

"""
2D velocity integration routines
"""

"""
Computes the integral of the 2D integrand, using the input wgts
"""
function integral(integrand::AbstractArray{mk_float,2},
                vx::Array{mk_float,1}, px::mk_int, wgtsx::Array{mk_float,1},
                vy::Array{mk_float,1}, py::mk_int, wgtsy::Array{mk_float,1})
    # nx is the number of grid points
    nx = length(wgtsx)
    ny = length(wgtsy)
    # initialize 'integral' to zero before sum
    integral = 0.0
    @boundscheck nx == size(integrand,1) || throw(BoundsError(integrand))
    @boundscheck ny == size(integrand,2) || throw(BoundsError(integrand))
    @boundscheck nx == length(vx) || throw(BoundsError(vx))
    @boundscheck ny == length(vy) || throw(BoundsError(vy))

    @inbounds for j ∈ 1:ny
        @inbounds for i ∈ 1:nx
            integral += integrand[i,j] * (vx[i] ^ px) * (vy[j] ^ py) * wgtsx[i] * wgtsy[j]
        end
    end
    return integral
end

end
