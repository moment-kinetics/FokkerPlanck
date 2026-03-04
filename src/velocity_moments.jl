"""
"""
module velocity_moments

export get_density
export get_upar
export get_ppar
export get_pperp
export get_qpar
export get_rmom
export get_pressure
export get_moment

using FiniteElementAssembly: FiniteElementCoordinate, integral

function get_density(ff::Tpdf,
            vpa::FiniteElementCoordinate,
            vperp::FiniteElementCoordinate) where Tpdf <: AbstractArray{Float64,2}
    # Integrating calculates n_s / nref = ∫d(vpa/cref) (f_s c_ref / N_e) in 1V
    # or n_s / nref = ∫d^3(v/cref) (f_s c_ref^3 / N_e) in 2V
    return integral((vpa,vperp)->(1.0), ff, vpa, vperp)
end

function get_upar(ff::Tpdf,
            vpa::FiniteElementCoordinate,
            vperp::FiniteElementCoordinate,
            density::Float64) where Tpdf <: AbstractArray{Float64,2}
    # Integrating calculates
    # (n_s / N_e) * (upar_s / c_s) = (1/√π)∫d(vpa/c_s) * (vpa/c_s) * (√π f_s c_s / N_e)
    # so we divide by the density of f_s
    upar = integral((vpa,vperp)->(vpa), ff, vpa, vperp)
    upar /= density
    return upar
end

function get_pperp(p::Float64, ppar::Float64)
    return 1.5 * p - 0.5 * ppar
end

function get_ppar(ff::Tpdf,
            vpa::FiniteElementCoordinate,
            vperp::FiniteElementCoordinate,
            upar::Float64, mass::Float64) where Tpdf <: AbstractArray{Float64,2}
    # Calculate ∫d^3v (vpa-upar)^2 ff
    return mass*integral((vpa,vperp)->((vpa-upar)^2), ff, vpa, vperp)
end

function get_pressure(ff::Tpdf,
            vpa::FiniteElementCoordinate,
            vperp::FiniteElementCoordinate,
            upar::Float64, mass::Float64) where Tpdf <: AbstractArray{Float64,2}
    # Integrating calculates
    # ∫d^3v (((vpa-upar))^2 + vperp^2) * ff
    return (mass/3.0)*integral((vpa,vperp)->((vpa - upar)^2 + vperp^2), ff, vpa, vperp)
end

function get_qpar(ff::Tpdf,
            vpa::FiniteElementCoordinate,
            vperp::FiniteElementCoordinate,
            upar::Float64, mass::Float64) where Tpdf <: AbstractArray{Float64,2}
    return 0.5 * mass *
            integral((vpa,vperp) -> (vpa-upar)*((vpa-upar)^2 + vperp^2), ff, vpa, vperp)
end

# generalised moment useful for computing numerical conserving terms in the collision operator
function get_rmom(ff::Tpdf,
            vpa::FiniteElementCoordinate,
            vperp::FiniteElementCoordinate,
            upar::Float64, mass::Float64) where Tpdf <: AbstractArray{Float64,2}
    return mass*integral((vpa,vperp)->((vpa-upar)^2 + vperp^2)^2, ff, vpa, vperp)
end

function get_nm_moment(n::Int64, m::Int64, ff::Tpdf,
            vpa::FiniteElementCoordinate,
            vperp::FiniteElementCoordinate) where Tpdf <: AbstractArray{Float64,2}
    return integral((vpa,vperp)->(vpa^n*vperp^m), ff, vpa, vperp)
end

end
