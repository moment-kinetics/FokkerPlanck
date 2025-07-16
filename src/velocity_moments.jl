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

using ..type_definitions: mk_float
using ..coordinates: finite_element_coordinate
using ..calculus: integral

function get_density(ff::AbstractArray{mk_float,2},
            vpa::finite_element_coordinate,
            vperp::finite_element_coordinate)
    # Integrating calculates n_s / nref = ∫d(vpa/cref) (f_s c_ref / N_e) in 1V
    # or n_s / nref = ∫d^3(v/cref) (f_s c_ref^3 / N_e) in 2V
    return integral(ff, vpa.grid, 0, vpa.wgts, vperp.grid, 0, vperp.wgts)
end

function get_upar(ff::AbstractArray{mk_float,2},
            vpa::finite_element_coordinate,
            vperp::finite_element_coordinate,
            density::mk_float)
    # Integrating calculates
    # (n_s / N_e) * (upar_s / c_s) = (1/√π)∫d(vpa/c_s) * (vpa/c_s) * (√π f_s c_s / N_e)
    # so we divide by the density of f_s
    upar = integral(ff, vpa.grid, 1, vpa.wgts, vperp.grid, 0, vperp.wgts)
    upar /= density
    return upar
end

function get_pperp(p::mk_float, ppar::mk_float)
    return 1.5 * p - 0.5 * ppar
end

function get_ppar(ff::AbstractArray{mk_float,2},
            vpa::finite_element_coordinate,
            vperp::finite_element_coordinate,
            upar::mk_float)
    # Calculate ∫d^3v (vpa-upar)^2 ff
    return integral((vperp,vpa)->((vpa-upar)^2), ff, vperp, vpa)
end

function get_pressure(ff::AbstractArray{mk_float,2},
            vpa::finite_element_coordinate,
            vperp::finite_element_coordinate,
            upar::mk_float)
    # Integrating calculates
    # ∫d^3v (((vpa-upar))^2 + vperp^2) * ff
    return (1.0/3.0)*integral((vperp,vpa)->((vpa - upar)^2 + vperp^2), ff, vperp, vpa)
end

function get_qpar(ff::AbstractArray{mk_float,2},
            vpa::finite_element_coordinate,
            vperp::finite_element_coordinate,
            upar::mk_float)
    return 0.5 *
            integral((vperp,vpa) -> (vpa-upar)*((vpa-upar)^2 + vperp^2), ff, vperp, vpa)
end

# generalised moment useful for computing numerical conserving terms in the collision operator
function get_rmom(ff::AbstractArray{mk_float,2},
            vpa::finite_element_coordinate,
            vperp::finite_element_coordinate,
            upar::mk_float)
    return integral((vperp,vpa)->((vpa-upar)^2 + vperp^2)^2, ff, vperp, vpa)
end

end
