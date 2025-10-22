using Dates
using FokkerPlanck.array_allocation: allocate_float
using FokkerPlanck.type_definitions: mk_float, mk_int
using FokkerPlanck: fokker_planck_backward_euler_data,
                    fokker_planck_collisions_backward_euler_step!,
                    fokker_planck_collision_operator_weak_form!,
                    fokkerplanck_weakform_arrays_struct,
                    multipole_expansion, delta_f_multipole, boundary_data_type,
                    multi_species_operator_type, single_assembly_per_species, repeat_assembly_per_species,
                    fixed_background_plasma_input, slowing_down_source_data_input
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
        +a^2*(y^5*exp(-y^2)/4 - 7*y^3*exp(-y^2)/8 - 13*y*exp(-y^2)/16 + 13*sqrt(pi)*erf(y)/32 - 13*sqrt(pi)/32)
        +a^3*(-y^8*exp(-y^2)/12 + 2*y^6*exp(-y^2)/3 + y^4*exp(-y^2)/2 + y^2*exp(-y^2) + exp(-y^2))
        +a^4*(y^11*exp(-y^2)/48 - 29*y^9*exp(-y^2)/96 + 9*y^7*exp(-y^2)/64 - y^5*exp(-y^2)/128 - 5*y^3*exp(-y^2)/256 - 15*y*exp(-y^2)/512 + 15*sqrt(pi)*erf(y)/1024 - 15*sqrt(pi)/1024)
        +a^5*(-y^14*exp(-y^2)/240 + 23*y^12*exp(-y^2)/240 - 31*y^10*exp(-y^2)/120 - 7*y^8*exp(-y^2)/24 - 7*y^6*exp(-y^2)/6 - 7*y^4*exp(-y^2)/2 - 7*y^2*exp(-y^2) - 7*exp(-y^2))
        +a^6*(y^17*exp(-y^2)/1440 - 67*y^15*exp(-y^2)/2880 + 53*y^13*exp(-y^2)/384 + 49*y^11*exp(-y^2)/768 + 923*y^9*exp(-y^2)/1536 + 2769*y^7*exp(-y^2)/1024 + 19383*y^5*exp(-y^2)/2048 + 96915*y^3*exp(-y^2)/4096 + 290745*y*exp(-y^2)/8192 - 290745*sqrt(pi)*erf(y)/16384 + 290745*sqrt(pi)/16384)
        +a^7*(y^18*exp(-y^2)/720 - y^16*exp(-y^2)/30 + y^14*exp(-y^2)/15 + y^12*exp(-y^2)/20 + 3*y^10*exp(-y^2)/10 + 3*y^8*exp(-y^2)/2 + 6*y^6*exp(-y^2) + 18*y^4*exp(-y^2) + 36*y^2*exp(-y^2) + 36*exp(-y^2))
        +a^8*(y^19*exp(-y^2)/1440 - 41*y^17*exp(-y^2)/2880 + 23*y^15*exp(-y^2)/5760 - 41*y^13*exp(-y^2)/768 - 533*y^11*exp(-y^2)/1536 - 5863*y^9*exp(-y^2)/3072 - 17589*y^7*exp(-y^2)/2048 - 123123*y^5*exp(-y^2)/4096 - 615615*y^3*exp(-y^2)/8192 - 1846845*y*exp(-y^2)/16384 + 1846845*sqrt(pi)*erf(y)/32768 - 1846845*sqrt(pi)/32768))
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
    sd_pdf = allocate_float(vpa.n,vperp.n,species.n)
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

function print_test_data(func_exact::AbstractArray{mk_float,2},
                    func_num::AbstractArray{mk_float,2},
                    func_err::AbstractArray{mk_float,2},
                    func_name::String,
                    vpa::finite_element_coordinate,
                    vperp::finite_element_coordinate,
                    dummy::AbstractArray{mk_float,2},
                    v_max::mk_float, v_min::mk_float;
                    print_to_screen=true::Bool)
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
    maxnorm::Vector{mk_float}
    L2norm::Vector{mk_float}
    maxnorm_relative::Vector{mk_float}
    L2norm_relative::Vector{mk_float}
    min_v::Vector{mk_float}
    max_v::Vector{mk_float}
    function SD_error_data(nspecies::mk_int)
        return new(allocate_float(nspecies),allocate_float(nspecies),allocate_float(nspecies),
            allocate_float(nspecies),allocate_float(nspecies),allocate_float(nspecies))
    end
end

function diagnose_F_SD!(sd_errors::SD_error_data,sd_pdf::AbstractArray{mk_float,3},
    Fold::AbstractArray{mk_float,3}, Fdummy1::AbstractArray{mk_float,3},
    Fdummy2::AbstractArray{mk_float,2}, Fdummy3::AbstractArray{mk_float,2},
    source_v0::Vector{mk_float}, vth_ion::mk_float,
    vpa::finite_element_coordinate, vperp::finite_element_coordinate, species::species_info;
    print_to_screen=true::Bool)
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