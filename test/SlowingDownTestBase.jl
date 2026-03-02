using Dates
using FokkerPlanck: FokkerPlanckBackwardEulerData,
                    fokker_planck_collisions_backward_euler_step!,
                    fokker_planck_collision_operator_weak_form!,
                    FokkerPlanckWeakformArrays,
                    multipole_expansion, delta_f_multipole, boundary_data_type,
                    multi_species_operator_type, single_assembly_per_species, repeat_assembly_per_species,
                    FixedBackgroundPlasmaInput, SlowingDownSourceInput
import FokkerPlanck.fokker_planck_test: print_test_data
using SpecialFunctions: erf

"""
Solution shape function due to a source of the form `exp( - (v^2 - v_b^2))^2/(4*v_th^2 v_b^2 ) )`.
"""
function shape_function_from_source(v,v0,vth)
    y = (v - v0)/vth
    a = vth/v0
    # shape = \int^\infty_y (1 + a x)^2 exp(-x^2) exp(-(a x^3 + 0.25 a^2 x^4))
    # we expand exp(-(a x^3 + 0.25 a^2 x^4)) to O(a^6) and compute the integral analytically
    # shape including O(a^8) terms
    shape = (+a^0*(-sqrt(pi)*erf(y)/2 + sqrt(pi)/2)
            +a^1*(-y^2*exp(-y^2)/2 + exp(-y^2)/2)
            +a^2*(y^5*exp(-y^2)/4 - y^3*exp(-y^2)/2 - y*exp(-y^2)/4 + sqrt(pi)*erf(y)/8 - sqrt(pi)/8)
            +a^3*(-y^8*exp(-y^2)/12 + 7*y^6*exp(-y^2)/24 + y^4*exp(-y^2)/8 + y^2*exp(-y^2)/4 + exp(-y^2)/4)
            +a^4*(y^11*exp(-y^2)/48 - 11*y^9*exp(-y^2)/96 - y^5*exp(-y^2)/8 - 5*y^3*exp(-y^2)/16 - 15*y*exp(-y^2)/32 + 15*sqrt(pi)*erf(y)/64 - 15*sqrt(pi)/64)
            +a^5*(-y^14*exp(-y^2)/240 + y^12*exp(-y^2)/30 - 23*y^10*exp(-y^2)/960 + 7*y^8*exp(-y^2)/192 + 7*y^6*exp(-y^2)/48 + 7*y^4*exp(-y^2)/16 + 7*y^2*exp(-y^2)/8 + 7*exp(-y^2)/8)
            +a^6*(y^17*exp(-y^2)/1440 - 11*y^15*exp(-y^2)/1440 + 5*y^13*exp(-y^2)/384 - y^11*exp(-y^2)/96 - y^9*exp(-y^2)/24 - 3*y^7*exp(-y^2)/16 - 21*y^5*exp(-y^2)/32 - 105*y^3*exp(-y^2)/64 - 315*y*exp(-y^2)/128 + 315*sqrt(pi)*erf(y)/256 - 315*sqrt(pi)/256)
            +a^7*(y^18*exp(-y^2)/720 - y^16*exp(-y^2)/480 + 19*y^14*exp(-y^2)/960 + 77*y^12*exp(-y^2)/640 + 231*y^10*exp(-y^2)/320 + 231*y^8*exp(-y^2)/64 + 231*y^6*exp(-y^2)/16 + 693*y^4*exp(-y^2)/16 + 693*y^2*exp(-y^2)/8 + 693*exp(-y^2)/8)
            +a^8*(y^19*exp(-y^2)/1440 + y^17*exp(-y^2)/720 + 113*y^15*exp(-y^2)/5760 + 7*y^13*exp(-y^2)/48 + 91*y^11*exp(-y^2)/96 + 1001*y^9*exp(-y^2)/192 + 3003*y^7*exp(-y^2)/128 + 21021*y^5*exp(-y^2)/256 + 105105*y^3*exp(-y^2)/512 + 315315*y*exp(-y^2)/1024 - 315315*sqrt(pi)*erf(y)/2048 + 315315*sqrt(pi)/2048)
            )
    # multiply Jacobian factor to convert dimensionless result to a velocity integral
    shape *= 4*pi*(v0^2*vth)
    return shape
end

"""
Analytical solution of slowing down problem for a source that is
isotropic in pitch angle and extended in v.
See [1,2] for solutions with isotropic sources that are delta functions in v.
The solution below due to an extended source can be constructed with the
Green's function approach for linear ODEs.

[1] Helander, P., & Sigmar D. J., (2002). Collisional Transport in Magnetized Plasmas, Cambridge University Press,  Chapter 3.4, pages 40-42
[2] Moseev, D., & Salewski, M. (2019). Bi-Maxwellian, slowing-down, and ring velocity distributions of fast ions in
magnetized plasmas. Physics of Plasmas, 26(2). https://doi.org/10.1063/1.5085429
"""
function slowing_down_pdf(vpa, vperp, species, source_rate, source_v0, source_vth,
        msp, Zsp, denssp, vthsp, nuref)
    sd_pdf = Array{Float64}(undef,vpa.n,vperp.n,species.n)
    ielectron = 1
    Te = 0.5*msp[ielectron]*vthsp[ielectron]^2
    me = msp[ielectron]
    ne = denssp[ielectron]
    for is in 1:species.n
        mf = species.mass[is]
        Zf = species.zeds[is]
        taus = (3.0/4.0)*sqrt(2.0*pi)*(mf/sqrt(me))*(Te^(1.5))*(Zf^(-2))*(1.0/ne)
        Z1 = 0.0
        for iion in 2:length(Zsp)
            ni = denssp[iion]
            mi = msp[iion]
            Zi = Zsp[iion]
            Z1 += (ni/ne)*(mf/mi)*(Zi^2)
        end
        vc3 = 3.0*sqrt(pi/2.0)*Z1*(Te^(1.5))/(mf*sqrt(me))
        amplitude = source_rate[is]*taus/(4.0*pi*nuref)
        v0 = source_v0[is]
        vth = source_vth[is]
        norm = shape_function_from_source(0.0,v0,vth)
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                v = (vpa.grid[ivpa]^2 + vperp.grid[ivperp]^2)^(0.5)
                shape = shape_function_from_source(v,v0,vth)/norm
                v3 = (vpa.grid[ivpa]^2 + vperp.grid[ivperp]^2)^(1.5)
                sd_pdf[ivpa,ivperp,is] = amplitude*shape/(vc3 + v3)
                # if source_v0[is]^3 > v3
                #     sd_pdf[ivpa,ivperp,is] = amplitude/(vc3 + v3)
                # else
                #     sd_pdf[ivpa,ivperp,is] = 0.0
                # end
            end
        end
    end
    return sd_pdf
end

function print_test_data(func_exact::Tpdf1,
                    func_num::Tpdf1,
                    func_err::Tpdf2,
                    func_name::String,
                    vpa::FiniteElementCoordinate,
                    vperp::FiniteElementCoordinate,
                    dummy::Tpdf2,
                    v_max::Float64, v_min::Float64;
                    print_to_screen=true::Bool
                    ) where {Tpdf1 <: AbstractArray{Float64,2}, Tpdf2 <: AbstractArray{Float64,2}}
    @. func_err = 0.0
    @. dummy = 0.0
    # compute error in range [v_min,v_max]
    for ivperp in 1:vperp.n
        for ivpa in 1:vpa.n
            v2 = vperp.grid[ivperp]^2 + vpa.grid[ivpa]^2
            if v2 < v_max^2 && v2 > v_min^2
                func_err[ivpa,ivperp] = abs(func_num[ivpa,ivperp] - func_exact[ivpa,ivperp])
                dummy[ivpa,ivperp] = func_err[ivpa,ivperp]^2
            end
        end
    end
    # compute the numerator
    num = get_density(dummy,vpa,vperp)
    # compute the denominator, volume of range in [v_min,v_max]
    for ivperp in 1:vperp.n
        for ivpa in 1:vpa.n
            v2 = vperp.grid[ivperp]^2 + vpa.grid[ivpa]^2
            if v2 < v_max^2 && v2 > v_min^2
                dummy[ivpa,ivperp] = 1.0
            end
        end
    end
    denom = get_density(dummy,vpa,vperp)
    L2norm = sqrt(num/denom)
    maxnorm = maximum(abs.(func_err))
    if print_to_screen
        println("maximum("*func_name*"): ",maxnorm," L2("*func_name*"): ",L2norm, " in v=[$v_min,$v_max]")
    end
    return maxnorm, L2norm
end

struct SD_error_data
    maxnorm::Vector{Float64}
    L2norm::Vector{Float64}
    maxnorm_relative::Vector{Float64}
    L2norm_relative::Vector{Float64}
    min_v::Vector{Float64}
    max_v::Vector{Float64}
    function SD_error_data(nspecies::Int64)
        return new(Array{Float64}(undef,nspecies),Array{Float64}(undef,nspecies),Array{Float64}(undef,nspecies),
            Array{Float64}(undef,nspecies),Array{Float64}(undef,nspecies),Array{Float64}(undef,nspecies))
    end
end

function diagnose_F_SD!(sd_errors::SD_error_data,sd_pdf::Tpdf1,
    Fold::Tpdf1, Fdummy1::Tpdf1,
    Fdummy2::Tpdf2, Fdummy3::Tpdf2,
    source_v0::Vector{Float64}, vth_ion::Float64,
    vpa::FiniteElementCoordinate, vperp::FiniteElementCoordinate, species::SpeciesData;
    print_to_screen=true::Bool) where {Tpdf1 <: AbstractArray{Float64,3}, Tpdf2 <: AbstractArray{Float64,2}} 
    @. Fdummy1 = 0.0
    for is in 1:species.n
        vmax = 1.1*source_v0[is]
        vmin = sqrt(source_v0[is]*2.0*vth_ion)
        @views maxnorm, L2norm = print_test_data(sd_pdf[:,:,is],
            Fold[:,:,is], Fdummy2, "|F[$is] - F_SD|", vpa, vperp,
            Fdummy3, vmax, vmin, print_to_screen=print_to_screen)
        @views maxF, L2F = print_test_data(Fdummy1[:,:,is],
            Fold[:,:,is], Fdummy2, "|F[$is]|", vpa, vperp,
            Fdummy3,  vmax, vmin, print_to_screen=false)
        if print_to_screen
            println("max(|F[$is] - F_SD|)/max(|F|): ", maxnorm/maxF," L2(|F[$is] - F_SD|)/L2(F): ",L2norm/L2F)
        end
        sd_errors.maxnorm[is] = maxnorm
        sd_errors.maxnorm_relative[is] = maxnorm/maxF
        sd_errors.L2norm[is] = L2norm
        sd_errors.L2norm_relative[is] = L2norm/L2F
        sd_errors.min_v[is] = vmin
        sd_errors.max_v[is] = vmax
    end
    return nothing
end