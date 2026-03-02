"""
Module for including functions used
in testing the implementation of the
the full-F Fokker-Planck collision operator.
"""
module fokker_planck_test

export Cflux_vpa_Maxwellian_inputs, Cflux_vperp_Maxwellian_inputs
export d2Gdvpa2_Maxwellian, dGdvperp_Maxwellian, d2Gdvperpdvpa_Maxwellian, d2Gdvperp2_Maxwellian
export dHdvpa_Maxwellian, dHdvperp_Maxwellian, Cssp_Maxwellian_inputs
export F_Maxwellian, dFdvpa_Maxwellian, dFdvperp_Maxwellian
export d2Fdvpa2_Maxwellian, d2Fdvperpdvpa_Maxwellian, d2Fdvperp2_Maxwellian
export H_Maxwellian, G_Maxwellian, F_Beam

export Cssp_fully_expanded_form, calculate_collisional_fluxes

export print_test_data, fkpl_error_data, allocate_error_data
export save_fkpl_error_data, save_fkpl_integration_error_data
using HDF5
using FiniteElementAssembly: FiniteElementCoordinate
using SpecialFunctions: erf
using ..velocity_moments: get_density
# below are a series of functions that can be used to test the calculation
# of the Rosenbluth potentials for a shifted Maxwellian
# or provide an estimate for collisional coefficients

"""
Function computing G, defined by
```math
\\nabla^4 G = -\\frac{8}{\\sqrt{\\pi}} F
```
with
```math
F = c_{\\rm ref}^3 F_{\\rm Maxwellian} / n_{\\rm ref}
```
the normalised Maxwellian.
See Plasma Confinement, R. D. Hazeltine & J. D. Meiss, 2003, Dover Publications, pg 184, Chpt 5.2, Eqn (5.49).
"""
function G_Maxwellian(dens::Float64,upar::Float64,vth::Float64,
                    vpa::Float64,vperp::Float64)
    # speed variable
    eta = eta_func(upar,vth,vpa,vperp)
    zero = 1.0e-10
    if eta < zero
        G = 2.0/sqrt(pi)
    else
        # G_M = (1/2 eta)*( eta erf'(eta) + (1 + 2 eta^2) erf(eta))
        G = (1.0/sqrt(pi))*exp(-eta^2) + ((0.5/eta) + eta)*erf(eta)
    end
    return G*dens*vth
end

"""
Function computing H, defined by
```math
\\nabla^2 H = -\\frac{4}{\\sqrt{\\pi}} F
```
with
```math
F = c_{\\rm ref}^3 F_{\\rm Maxwellian} / n_{\\rm ref}
```
the normalised Maxwellian.
See Plasma Confinement, R. D. Hazeltine & J. D. Meiss, 2003, Dover Publications, pg 184, Chpt 5.2, Eqn (5.49).
"""
function H_Maxwellian(dens::Float64,upar::Float64,vth::Float64,
                    vpa::Float64,vperp::Float64)
    # speed variable
    eta = eta_func(upar,vth,vpa,vperp)
    zero = 1.0e-10
    if eta < zero
        # erf(eta)/eta ~ 2/sqrt(π) + O(eta^2) for eta << 1
        H = 2.0/sqrt(pi)
    else
        # H_M =  erf(eta)/eta
        H = erf(eta)/eta
    end
    return H*dens/vth
end

# 1D derivative functions

function dGdeta(eta::Float64)
    # d \tilde{G} / d eta
    dGdeta_fac = (1.0/sqrt(pi))*exp(-eta^2)/eta + (1.0 - 0.5/(eta^2))*erf(eta)
    return dGdeta_fac
end

function d2Gdeta2(eta::Float64)
    # d \tilde{G} / d eta
    d2Gdeta2_fac = erf(eta)/(eta^3) - (2.0/sqrt(pi))*exp(-eta^2)/(eta^2)
    return d2Gdeta2_fac
end

function ddGddeta(eta::Float64)
    # d / d eta ( (1/ eta) d \tilde{G} d eta
    ddGddeta_fac = (1.5/(eta^2) - 1.0)*erf(eta)/(eta^2) - (3.0/sqrt(pi))*exp(-eta^2)/(eta^3)
    return ddGddeta_fac
end

function dHdeta(eta::Float64)
    dHdeta_fac = (2.0/sqrt(pi))*(exp(-eta^2))/eta - erf(eta)/(eta^2)
    return dHdeta_fac
end

"""
Function computing the normalised speed variable
```math
\\eta = \\frac{\\sqrt{(v_\\| - u_\\|)^2 + v_\\perp^2}}{v_{\\rm th}}
```
with \$v_{\\rm th} = \\sqrt{2 p / n m}\$ the thermal speed, and \$p\$ the pressure,
 \$n\$ the density and \$m\$ the mass.
"""
function eta_func(upar::Float64,vth::Float64,
             vpa::Float64,vperp::Float64)
    speed = sqrt( (vpa - upar)^2 + vperp^2)/vth
    return speed
end

"""
Function computing
```math
\\frac{\\partial^2 G }{ \\partial v_\\|^2}
```
 for Maxwellian input. See `G_Maxwellian()`.
"""
function d2Gdvpa2_Maxwellian(dens::Float64,upar::Float64,vth::Float64,
                            vpa::Float64,vperp::Float64)
    eta = eta_func(upar,vth,vpa,vperp)
    fac = dGdeta(eta) + ddGddeta(eta)*((vpa - upar)^2)/(vth^2)
    d2Gdvpa2_fac = fac*dens/(eta*vth)
    return d2Gdvpa2_fac
end

"""
Function computing
```math
\\frac{\\partial^2 G}{\\partial v_\\perp \\partial v_\\|}
```
for Maxwellian input. See `G_Maxwellian()`.
"""
function d2Gdvperpdvpa_Maxwellian(dens::Float64,upar::Float64,vth::Float64,
                            vpa::Float64,vperp::Float64)
    eta = eta_func(upar,vth,vpa,vperp)
    fac = ddGddeta(eta)*vperp*(vpa - upar)/(vth^2)
    d2Gdvperpdvpa_fac = fac*dens/(eta*vth)
    return d2Gdvperpdvpa_fac
end

"""
Function computing
```math
\\frac{\\partial^2 G}{\\partial v_\\perp^2}
```
for Maxwellian input. See `G_Maxwellian()`.
"""
function d2Gdvperp2_Maxwellian(dens::Float64,upar::Float64,vth::Float64,
                            vpa::Float64,vperp::Float64)
    eta = eta_func(upar,vth,vpa,vperp)
    fac = dGdeta(eta) + ddGddeta(eta)*(vperp^2)/(vth^2)
    d2Gdvperp2_fac = fac*dens/(eta*vth)
    return d2Gdvperp2_fac
end

"""
Function computing
```math
\\frac{\\partial G}{\\partial v_\\perp}
```
for Maxwellian input. See `G_Maxwellian()`.
"""
function dGdvperp_Maxwellian(dens::Float64,upar::Float64,vth::Float64,
                            vpa::Float64,vperp::Float64)
    eta = eta_func(upar,vth,vpa,vperp)
    fac = dGdeta(eta)*vperp*dens/(vth*eta)
    return fac
end

"""
Function computing
```math
\\frac{\\partial H}{\\partial v_\\perp}
```
for Maxwellian input. See `H_Maxwellian()`.
"""
function dHdvperp_Maxwellian(dens::Float64,upar::Float64,vth::Float64,
                            vpa::Float64,vperp::Float64)
    eta = eta_func(upar,vth,vpa,vperp)
    fac = dHdeta(eta)*vperp*dens/(eta*vth^3)
    return fac
end

"""
Function computing
```math
\\frac{\\partial H}{\\partial v_\\|}
```
for Maxwellian input. See `H_Maxwellian()`.
"""
function dHdvpa_Maxwellian(dens::Float64,upar::Float64,vth::Float64,
                            vpa::Float64,vperp::Float64)
    eta = eta_func(upar,vth,vpa,vperp)
    fac = dHdeta(eta)*(vpa-upar)*dens/(eta*vth^3)
    return fac
end

"""
Function computing \$ F_{\\rm Maxwellian} \$.
"""
function F_Maxwellian(dens::Float64,upar::Float64,vth::Float64,
                        vpa::Float64,vperp::Float64)
    eta = eta_func(upar,vth,vpa,vperp)
    fac = (dens/(vth^3)/π^1.5)*exp(-eta^2)
    return fac
end

"""
Function computing \$ F_{\\rm Beam} \$.
"""
function F_Beam(vpa0::Float64,vperp0::Float64,vth0::Float64,
                        vpa::Float64,vperp::Float64)
    w2 = (vpa-vpa0)^2 + (vperp-vperp0)^2
    fac = exp(-(w2)/(vth0^2))
    return fac
end

"""
Function computing
```math
\\frac{\\partial F}{\\partial v_\\|}
```
for \$ F = F_{\\rm Maxwellian}\$.
"""
function dFdvpa_Maxwellian(dens::Float64,upar::Float64,vth::Float64,
                            vpa::Float64,vperp::Float64)
    eta = eta_func(upar,vth,vpa,vperp)
    fac = -2.0*(dens/(vth^4)/π^1.5)*((vpa - upar)/vth)*exp(-eta^2)
    return fac
end

"""
Function computing
```math
\\frac{\\partial F}{\\partial v_\\perp}
```
for \$ F = F_{\\rm Maxwellian}\$.
"""
function dFdvperp_Maxwellian(dens::Float64,upar::Float64,vth::Float64,
                            vpa::Float64,vperp::Float64)
    eta = eta_func(upar,vth,vpa,vperp)
    fac = -2.0*(dens/(vth^4)/π^1.5)*(vperp/vth)*exp(-eta^2)
    return fac
end

"""
Function computing
```math
\\frac{\\partial^2 F}{\\partial v_\\perp \\partial v_\\|}
```
for \$ F = F_{\\rm Maxwellian}\$.
"""
function d2Fdvperpdvpa_Maxwellian(dens::Float64,upar::Float64,vth::Float64,
                            vpa::Float64,vperp::Float64)
    eta = eta_func(upar,vth,vpa,vperp)
    fac = 4.0*(dens/(vth^5)/π^1.5)*(vperp/vth)*((vpa - upar)/vth)*exp(-eta^2)
    return fac
end

"""
Function computing
```math
\\frac{\\partial^2 F}{\\partial v_\\|^2}
```
for \$ F = F_{\\rm Maxwellian}\$.
"""
function d2Fdvpa2_Maxwellian(dens::Float64,upar::Float64,vth::Float64,
                            vpa::Float64,vperp::Float64)
    eta = eta_func(upar,vth,vpa,vperp)
    fac = 4.0*(dens/(vth^5)/π^1.5)*( ((vpa - upar)/vth)^2 - 0.5 )*exp(-eta^2)
    return fac
end

"""
Function computing
```math
\\frac{\\partial^2 F}{\\partial v_\\perp^2}.
```
for \$ F = F_{\\rm Maxwellian}\$.
"""
function d2Fdvperp2_Maxwellian(dens::Float64,upar::Float64,vth::Float64,
                            vpa::Float64,vperp::Float64)
    eta = eta_func(upar,vth,vpa,vperp)
    fac = 4.0*(dens/(vth^5)/π^1.5)*((vperp/vth)^2 - 0.5)*exp(-eta^2)
    return fac
end

"""
Calculates the fully expanded form of the collision operator \$C_{s s^\\prime}[F_s,F_{s^\\prime}]\$ given Maxwellian input \$F_s\$ and \$F_{s^\\prime}\$.
The input Maxwellians are specified through their moments.
"""
function Cssp_Maxwellian_inputs(denss::Float64,upars::Float64,vths::Float64,ms::Float64,Zs::Float64,
                                denssp::Float64,uparsp::Float64,vthsp::Float64,msp::Float64,Zsp::Float64,
                                nussp::Float64,vpa::Float64,vperp::Float64)

    d2Fsdvpa2 = d2Fdvpa2_Maxwellian(denss,upars,vths,vpa,vperp)
    d2Fsdvperp2 = d2Fdvperp2_Maxwellian(denss,upars,vths,vpa,vperp)
    d2Fsdvperpdvpa = d2Fdvperpdvpa_Maxwellian(denss,upars,vths,vpa,vperp)
    dFsdvperp = dFdvperp_Maxwellian(denss,upars,vths,vpa,vperp)
    dFsdvpa = dFdvpa_Maxwellian(denss,upars,vths,vpa,vperp)
    Fs = F_Maxwellian(denss,upars,vths,vpa,vperp)

    d2Gspdvpa2 = d2Gdvpa2_Maxwellian(denssp,uparsp,vthsp,vpa,vperp)
    d2Gspdvperp2 = d2Gdvperp2_Maxwellian(denssp,uparsp,vthsp,vpa,vperp)
    d2Gspdvperpdvpa = d2Gdvperpdvpa_Maxwellian(denssp,uparsp,vthsp,vpa,vperp)
    dGspdvperp = dGdvperp_Maxwellian(denssp,uparsp,vthsp,vpa,vperp)
    dHspdvperp = dHdvperp_Maxwellian(denssp,uparsp,vthsp,vpa,vperp)
    dHspdvpa = dHdvpa_Maxwellian(denssp,uparsp,vthsp,vpa,vperp)
    Fsp = F_Maxwellian(denssp,uparsp,vthsp,vpa,vperp)

    ( Cssp_Maxwellian =
        d2Fsdvpa2*d2Gspdvpa2 +
        d2Fsdvperp2*d2Gspdvperp2 +
        2.0*d2Fsdvperpdvpa*d2Gspdvperpdvpa +
        (1.0/(vperp^2))*dFsdvperp*dGspdvperp +
        2.0*(1.0 - (ms/msp))*(dFsdvpa*dHspdvpa + dFsdvperp*dHspdvperp) +
        (8.0*pi)*(ms/msp)*Fs*Fsp)
    # gamma_ss' = 2 pi e^4 ln \Lambda_ss' / (4 pi \epsilon_0)^2
    # nussp = (Lref/cref) (gamma_ss' nref / mref^2 cref^3)
    Cssp_Maxwellian *= nussp*(Zs*Zsp/ms)^2
    return Cssp_Maxwellian
end

"""
Calculates the collisional flux \$\\Gamma_\\|\$ given Maxwellian input \$F_s\$ and \$F_{s^\\prime}\$.
The input Maxwellians are specified through their moments.
"""
function Cflux_vpa_Maxwellian_inputs(ms::Float64,denss::Float64,upars::Float64,vths::Float64,
                                     msp::Float64,denssp::Float64,uparsp::Float64,vthsp::Float64,
                                     vpa::Float64,vperp::Float64)
    etap = eta_func(uparsp,vthsp,vpa,vperp)
    eta = eta_func(upars,vths,vpa,vperp)
    prefac = -2.0*denss*denssp*exp( -eta^2)/(vthsp*vths^5)
    (fac = (vpa-uparsp)*(d2Gdeta2(etap) + (ms/msp)*((vths/vthsp)^2)*dHdeta(etap)/etap)
             + (uparsp - upars)*( dGdeta(etap) + ((vpa-uparsp)^2/vthsp^2)*ddGddeta(etap) )/etap )
    Cflux = prefac*fac
    #fac *= (ms/msp)*(vths/vthsp)*dHdeta(etap)/etap
    #fac *= d2Gdeta2(etap)
    return Cflux
end

"""
Calculates the collisional flux \$\\Gamma_\\perp\$ given Maxwellian input \$F_s\$ and \$F_{s^\\prime}\$.
The input Maxwellians are specified through their moments.
"""
function Cflux_vperp_Maxwellian_inputs(ms::Float64,denss::Float64,upars::Float64,vths::Float64,
                                     msp::Float64,denssp::Float64,uparsp::Float64,vthsp::Float64,
                                     vpa::Float64,vperp::Float64)
    etap = eta_func(uparsp,vthsp,vpa,vperp)
    eta = eta_func(upars,vths,vpa,vperp)
    prefac = -2.0*(vperp)*denss*denssp*exp( -eta^2)/(vthsp*vths^5)
    (fac = (d2Gdeta2(etap) + (ms/msp)*((vths/vthsp)^2)*dHdeta(etap)/etap)
             + ((uparsp - upars)*(vpa-uparsp)/vthsp^2)*ddGddeta(etap)/etap )
    Cflux = prefac*fac
    #fac *= (ms/msp)*(vths/vthsp)*dHdeta(etap)/etap
    #fac *= d2Gdeta2(etap)
    return Cflux
end

"""
Function calculating the fully expanded form of the collision operator
taking as arguments the derivatives of \$F_s\$, \$G_{s^\\prime}\$ and \$H_{s^\\prime}\$.
This function is designed to be used at the
lowest level of a coordinate loop, with derivatives and integrals
all previously calculated.
"""
function Cssp_fully_expanded_form(nussp::Float64,ms::Float64,msp::Float64,
            d2fsdvpa2::Float64,d2fsdvperp2::Float64,d2fsdvperpdvpa::Float64,
            dfsdvpa::Float64,dfsdvperp::Float64,fs::Float64,
            d2Gspdvpa2::Float64,d2Gspdvperp2::Float64,d2Gspdvperpdvpa::Float64,
            dGspdvperp::Float64,dHspdvpa::Float64,dHspdvperp::Float64,
            fsp::Float64,vperp_val::Float64)
    ( Cssp = nussp*( d2fsdvpa2*d2Gspdvpa2 +
              d2fsdvperp2*d2Gspdvperp2 +
              2.0*d2fsdvperpdvpa*d2Gspdvperpdvpa +
              (1.0/(vperp_val^2))*dfsdvperp*dGspdvperp +
              2.0*(1.0 - (ms/msp))*(dfsdvpa*dHspdvpa + dfsdvperp*dHspdvperp) +
              (8.0/sqrt(pi))*(ms/msp)*fs*fsp) )
    return Cssp
end


"""
Calculates the collisional fluxes given input \$F_s\$ and \$G_{s^\\prime}\$, \$H_{s^\\prime}\$.
"""
function calculate_collisional_fluxes(F::Float64,dFdvpa::Float64,dFdvperp::Float64,
                            d2Gdvpa2::Float64,d2Gdvperpdvpa::Float64,d2Gdvperp2::Float64,
                            dHdvpa::Float64,dHdvperp::Float64,ms::Float64,msp::Float64)
    # fill in value at (ivpa,ivperp)
    Cflux_vpa = dFdvpa*d2Gdvpa2 + dFdvperp*d2Gdvperpdvpa - 2.0*(ms/msp)*F*dHdvpa
    #Cflux_vpa = dFdvpa*d2Gdvpa2 + dFdvperp*d2Gdvperpdvpa # - 2.0*(ms/msp)*F*dHdvpa
    #Cflux_vpa =  - 2.0*(ms/msp)*F*dHdvpa
    Cflux_vperp = dFdvpa*d2Gdvperpdvpa + dFdvperp*d2Gdvperp2 - 2.0*(ms/msp)*F*dHdvperp
    return Cflux_vpa, Cflux_vperp
end


# Below are functions which are used for storing and printing data from the tests

"""
Function to print the maximum error \${\\rm MAX}(|f_{\\rm numerical}-f_{\\rm exact}|)\$.
"""
function print_test_data(func_exact::Tpdf1,
            func_num::Tpdf2,
            func_err::Tpdf3,
            func_name::String) where {Tpdf <: AbstractArray{Float64,2},
                Tpdf1 <: Tpdf, Tpdf2 <: Tpdf, Tpdf3 <: Tpdf}
    @. func_err = abs(func_num - func_exact)
    max_err = maximum(func_err)
    println("maximum("*func_name*"_err): ",max_err)
    return max_err
end

"""
Function to print the maximum error \${\\rm MAX}(|f_{\\rm numerical}-f_{\\rm exact}|)\$ and the
\$L_2\$ norm of the error
```math
\\sqrt{\\int (f - f_{\\rm exact})^2 v_\\perp d v_\\perp d v_\\|/\\int v_\\perp d v_\\perp d v_\\|}.
```
"""
function print_test_data(func_exact::Tpdf1,
            func_num::Tpdf2,
            func_err::Tpdf3,
            func_name::String,
            vpa::FiniteElementCoordinate,
            vperp::FiniteElementCoordinate,
            dummy::Tpdf4; print_to_screen=true::Bool
            ) where {Tpdf <: AbstractArray{Float64,2},
            Tpdf1 <: Tpdf, Tpdf2 <: Tpdf, Tpdf3 <: Tpdf, Tpdf4 <: Tpdf}
    @. func_err = abs(func_num - func_exact)
    max_err = maximum(func_err)
    @. dummy = func_err^2
    # compute the numerator
    num = get_density(dummy,vpa,vperp)
    # compute the denominator
    @. dummy = 1.0
    denom = get_density(dummy,vpa,vperp)
    L2norm = sqrt(num/denom)
    if print_to_screen
        println("maximum("*func_name*"_err): ",max_err," L2("*func_name*"_err): ",L2norm)
    end
    return max_err, L2norm
end

mutable struct error_data
    max::Float64
    L2::Float64
end

mutable struct moments_error_data
    delta_density::Float64
    delta_upar::Float64
    delta_pressure::Float64
end

struct fkpl_error_data
    C_M::error_data
    H_M::error_data
    dHdvpa_M::error_data
    dHdvperp_M::error_data
    G_M::error_data
    dGdvperp_M::error_data
    d2Gdvpa2_M::error_data
    d2Gdvperpdvpa_M::error_data
    d2Gdvperp2_M::error_data
    moments::moments_error_data
end

function allocate_error_data()
    C_M = error_data(0.0,0.0)
    H_M = error_data(0.0,0.0)
    dHdvpa_M = error_data(0.0,0.0)
    dHdvperp_M = error_data(0.0,0.0)
    G_M = error_data(0.0,0.0)
    dGdvperp_M = error_data(0.0,0.0)
    d2Gdvpa2_M = error_data(0.0,0.0)
    d2Gdvperpdvpa_M = error_data(0.0,0.0)
    d2Gdvperp2_M = error_data(0.0,0.0)
    moments = moments_error_data(0.0,0.0,0.0)
    return fkpl_error_data(C_M,H_M,dHdvpa_M,dHdvperp_M,
        G_M,dGdvperp_M,d2Gdvpa2_M,d2Gdvperpdvpa_M,d2Gdvperp2_M,
        moments)
end

"""
Utility function that saves error data to a HDF5 file for later use.
"""
function save_fkpl_error_data(outdir::String,ncore::Int64,ngrid::Int64,nelement_list::Vector{Int64},
    max_C_err::Vector{Float64}, max_H_err::Vector{Float64}, max_G_err::Vector{Float64}, max_dHdvpa_err::Vector{Float64}, max_dHdvperp_err::Vector{Float64},
    max_d2Gdvperp2_err::Vector{Float64}, max_d2Gdvpa2_err::Vector{Float64}, max_d2Gdvperpdvpa_err::Vector{Float64}, max_dGdvperp_err::Vector{Float64},
    L2_C_err::Vector{Float64}, L2_H_err::Vector{Float64}, L2_G_err::Vector{Float64}, L2_dHdvpa_err::Vector{Float64}, L2_dHdvperp_err::Vector{Float64}, L2_d2Gdvperp2_err::Vector{Float64},
    L2_d2Gdvpa2_err::Vector{Float64}, L2_d2Gdvperpdvpa_err::Vector{Float64}, L2_dGdvperp_err::Vector{Float64},
    n_err::Vector{Float64}, u_err::Vector{Float64}, p_err::Vector{Float64}, calculate_times::Vector{Float64}, init_times::Vector{Float64}, expected_t_2::Vector{Float64}, expected_t_3::Vector{Float64},
    expected_diff::Vector{Float64}, expected_integral::Vector{Float64})
    filename = outdir*"fkpl_error_data_ngrid_"*string(ngrid)*"_ncore_"*string(ncore)*".h5"
    fid = h5open(filename, "w")
    fid["ncore"] = ncore
    fid["ngrid"] = ngrid
    fid["nelement_list"] = nelement_list
    fid["max_C_err"] = max_C_err
    fid["max_H_err"] = max_H_err
    fid["max_G_err"] = max_G_err
    fid["max_dHdvpa_err"] = max_dHdvpa_err
    fid["max_dHdvperp_err"] = max_dHdvperp_err
    fid["max_d2Gdvperp2_err"] = max_d2Gdvperp2_err
    fid["max_d2Gdvpa2_err"] = max_d2Gdvpa2_err
    fid["max_d2Gdvperpdvpa_err"] = max_d2Gdvperpdvpa_err
    fid["max_dGdvperp_err"] = max_dGdvperp_err
    fid["L2_C_err"] = L2_C_err
    fid["L2_H_err"] = L2_H_err
    fid["L2_G_err"] = L2_G_err
    fid["L2_dHdvpa_err"] = L2_dHdvpa_err
    fid["L2_dHdvperp_err"] = L2_dHdvperp_err
    fid["L2_d2Gdvperp2_err"] = L2_d2Gdvperp2_err
    fid["L2_d2Gdvpa2_err"] = L2_d2Gdvpa2_err
    fid["L2_d2Gdvperpdvpa_err"] = L2_d2Gdvperpdvpa_err
    fid["L2_dGdvperp_err"] = L2_dGdvperp_err
    fid["n_err"] = n_err
    fid["u_err"] = u_err
    fid["p_err"] = p_err
    fid["calculate_times"] = calculate_times
    fid["init_times"] = init_times
    fid["expected_t_2"] = expected_t_2
    fid["expected_t_3"] = expected_t_3
    fid["expected_diff"] = expected_diff
    fid["expected_integral"] = expected_integral
    close(fid)
    println("Saving error data: ",filename)
    return nothing
end

"""
Utility function that saves error data to a HDF5 file for later use.
"""
function save_fkpl_integration_error_data(outdir::String,ncore::Int64,ngrid::Int64,nelement_list::Vector{Int64},
    max_dfsdvpa_err::Vector{Float64}, max_dfsdvperp_err::Vector{Float64}, max_d2fsdvperpdvpa_err::Vector{Float64},
    max_H_err::Vector{Float64}, max_G_err::Vector{Float64}, max_dHdvpa_err::Vector{Float64}, max_dHdvperp_err::Vector{Float64},
    max_d2Gdvperp2_err::Vector{Float64}, max_d2Gdvpa2_err::Vector{Float64}, max_d2Gdvperpdvpa_err::Vector{Float64}, max_dGdvperp_err::Vector{Float64},
    expected_diff::Vector{Float64}, expected_integral::Vector{Float64})
    filename = outdir*"fkpl_integration_error_data_ngrid_"*string(ngrid)*"_ncore_"*string(ncore)*".h5"
    fid = h5open(filename, "w")
    fid["ncore"] = ncore
    fid["ngrid"] = ngrid
    fid["nelement_list"] = nelement_list
    fid["max_dfsdvpa_err"] = max_dfsdvpa_err
    fid["max_dfsdvperp_err"] = max_dfsdvperp_err
    fid["max_d2fsdvperpdvpa_err"] = max_d2fsdvperpdvpa_err
    fid["max_H_err"] = max_H_err
    fid["max_G_err"] = max_G_err
    fid["max_dHdvpa_err"] = max_dHdvpa_err
    fid["max_dHdvperp_err"] = max_dHdvperp_err
    fid["max_d2Gdvperp2_err"] = max_d2Gdvperp2_err
    fid["max_d2Gdvpa2_err"] = max_d2Gdvpa2_err
    fid["max_d2Gdvperpdvpa_err"] = max_d2Gdvperpdvpa_err
    fid["max_dGdvperp_err"] = max_dGdvperp_err
    fid["expected_diff"] = expected_diff
    fid["expected_integral"] = expected_integral
    close(fid)
    println("Saving error data: ",filename)
    return nothing
end

end
