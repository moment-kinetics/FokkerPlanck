"""
Package for computing the Full-F Fokker-Planck Collision Operator.

We implement the multi-species Collision operator
using the Rosenbluth-MacDonald-Judd formulation in a divergence form.
The Rosenbluth potentials are found using Poisson solvers
with boundary data supplied either from multipole expansions
or direct integration of the Rosenbluth potential definitions.
Higher-order finite element methods are used, with a projection
onto a weak-form using a nodal basis.
Results are returned evaluated on collocation points.

An implicit backward Euler solver with time-lagged preconditioner
is provided for testing purposes.

Documentation of methods can be found in the following publication.

M.R. Hardman, M. Abazorius, J. Omotani, M. Barnes, S.L. Newton, J.W.S. Cook, P.E. Farrell, F.I. Parra,
A higher-order finite-element implementation of the nonlinear Fokker--Planck collision operator for charged particle collisions in a low density plasma,
Computer Physics Communications, Volume 314, 2025, 109675,
https://doi.org/10.1016/j.cpc.2025.109675

"""
module FokkerPlanck

# Include submodules from other source files
# Note that order of includes matters - things used in one module must already
# be defined
include("type_definitions.jl")
include("array_allocation.jl")
include("coordinates.jl")
include("calculus.jl")
include("velocity_moments.jl")
include("fokker_planck_test.jl")
include("fokker_planck_nonlinear_solvers.jl")
include("fokker_planck_calculus.jl")

export fokker_planck_collision_operator_weak_form!
# entropy diagnostic
export calculate_entropy_production
# implicit advance
export fokker_planck_collisions_backward_euler_step!
# fixed background plasma inputs
export fixed_background_plasma_input
# source inputs
export slowing_down_source_data_input

using Dates
using LinearAlgebra: lu, ldiv!
using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float
using ..velocity_moments: get_density, get_upar, get_pressure
using ..fokker_planck_calculus: fokkerplanck_weakform_arrays_struct, fokker_planck_backward_euler_data,
                                fokker_planck_collision_operator_solve!,
                                enforce_vpavperp_BCs!,
                                calculate_rosenbluth_potentials_via_elliptic_solve!,
                                calculate_rosenbluth_potentials_via_analytical_Maxwellian!,
                                calculate_test_particle_preconditioner!,
                                advance_linearised_test_particle_collisions!,
                                multipole_expansion, direct_integration, delta_f_multipole, boundary_data_type,
                                conserving_corrections!, density_conserving_correction!,
                                species_info, calculate_cross_species_rosenbluth_potential_sums!,
                                multi_species_operator_type, single_assembly_per_species, repeat_assembly_per_species,
                                fixed_background_plasma_input, slowing_down_source_data_input,
                                slowing_down_source!, slowing_down_sink!, add_slowing_down_source!
using ..fokker_planck_test: d2Gdvpa2_Maxwellian, d2Gdvperpdvpa_Maxwellian, d2Gdvperp2_Maxwellian, dHdvpa_Maxwellian, dHdvperp_Maxwellian,
                            F_Maxwellian, dFdvpa_Maxwellian, dFdvperp_Maxwellian
using JacobianFreeNewtonKrylov: newton_solve!
# use and export ElementCoordinates so that users of the
# coordinate definition interface do not have to
# also know to install FiniteElementMatrices
using FiniteElementMatrices: ElementCoordinates
export ElementCoordinates
"""
Function for evaluating \$C_{ss'} = C_{ss'}[F_s,F_{s'}]\$

The result is stored in the array `fkpl_arrays.CC`.

The normalised collision frequency for collisions between species s and s' is defined by
```math
\\tilde{\\nu}_{ss'} = \\frac{L_{\\mathrm{ref}}}{c_{\\mathrm{ref}}}\\frac{\\gamma_{ss'} n_\\mathrm{ref}}{m_s^2 c_\\mathrm{ref}^3}
```
with \$\\gamma_{ss'} = 2 \\pi (Z_s Z_{s'})^2 e^4 \\ln \\Lambda_{ss'} / (4 \\pi
\\epsilon_0)^2\$.
The input parameter to this code is
```math
\\tilde{\\nu}_{ii} = \\frac{L_{\\mathrm{ref}}}{c_{\\mathrm{ref}}}\\frac{\\gamma_\\mathrm{ref} n_\\mathrm{ref}}{m_\\mathrm{ref}^2 c_\\mathrm{ref}^3}
```
with \$\\gamma_\\mathrm{ref} = 2 \\pi e^4 \\ln \\Lambda_{ii} / (4 \\pi
\\epsilon_0)^2\$. This means that \$\\tilde{\\nu}_{ss'} = (Z_s Z_{s'})^2\\tilde{\\nu}_\\mathrm{ref}\$ and this conversion is handled explicitly in the code with the charge number input provided by the user.
"""
function fokker_planck_collision_operator_weak_form!(
                         CCssp::AbstractArray{mk_float,2},
                         ffs_in::AbstractArray{mk_float,2},
                         ffsp_in::AbstractArray{mk_float,2},
                         ms::mk_float, msp::mk_float, nussp::mk_float,
                         fkpl_arrays::fokkerplanck_weakform_arrays_struct;
                         use_Maxwellian_Rosenbluth_coefficients=false::Bool,
                         algebraic_solve_for_d2Gdvperp2 = false::Bool, calculate_GG=false::Bool,
                         calculate_dGdvperp=false::Bool)
    # extract coordinates for boundscheck
    vpa = fkpl_arrays.vpa
    vperp = fkpl_arrays.vperp
    @boundscheck vpa.n == size(ffsp_in,1) || throw(BoundsError(ffsp_in))
    @boundscheck vperp.n == size(ffsp_in,2) || throw(BoundsError(ffsp_in))
    @boundscheck vpa.n == size(ffs_in,1) || throw(BoundsError(ffs_in))
    @boundscheck vperp.n == size(ffs_in,2) || throw(BoundsError(ffs_in))

    # extract the necessary precalculated and buffer arrays from fokkerplanck_arrays
    rhsvpavperp = fkpl_arrays.fprp_solver_data.matrix_operators.rhsvpavperp
    lu_obj_MM = fkpl_arrays.fprp_solver_data.matrix_operators.lu_obj_MM
    YY_arrays = fkpl_arrays.YY_arrays
    rosenbluth_potentials = fkpl_arrays.rosenbluth_potentials
    if use_Maxwellian_Rosenbluth_coefficients
        calculate_rosenbluth_potentials_via_analytical_Maxwellian!(rosenbluth_potentials,ffsp_in,vpa,vperp,msp)
    else
        calculate_rosenbluth_potentials_via_elliptic_solve!(rosenbluth_potentials,ffsp_in,
             vpa,vperp,fkpl_arrays.fprp_solver_data,msp,
             algebraic_solve_for_d2Gdvperp2=algebraic_solve_for_d2Gdvperp2,
             calculate_GG=calculate_GG,calculate_dGdvperp=calculate_dGdvperp)
    end
    # assemble weak form and solve mass matrix problem for CCssp
    fokker_planck_collision_operator_solve!(
                         CCssp, ffs_in, rosenbluth_potentials, ms, msp, nussp,
                         rhsvpavperp, lu_obj_MM, YY_arrays, vpa, vperp)
    return nothing
end
function fokker_planck_collision_operator_weak_form!(
                         CCs::AbstractArray{mk_float,3},
                         ff_in::AbstractArray{mk_float,3},
                         nuref::mk_float,
                         fkpl_arrays::fokkerplanck_weakform_arrays_struct;
                         use_conserving_corrections=false::Bool,
                         use_Maxwellian_Rosenbluth_coefficients=false::Bool,
                         algebraic_solve_for_d2Gdvperp2 = false::Bool, calculate_GG=false::Bool,
                         calculate_dGdvperp=false::Bool)
    # extract coordinates for boundscheck
    vpa = fkpl_arrays.vpa
    vperp = fkpl_arrays.vperp
    species = fkpl_arrays.species
    @boundscheck vpa.n == size(ff_in,1) || throw(BoundsError(ff_in))
    @boundscheck vperp.n == size(ff_in,2) || throw(BoundsError(ff_in))
    @boundscheck species.n == size(ff_in,3) || throw(BoundsError(ff_in))

    # extract the necessary precalculated and buffer arrays from fokkerplanck_arrays
    lu_obj_MM = fkpl_arrays.fprp_solver_data.matrix_operators.lu_obj_MM
    YY_arrays = fkpl_arrays.YY_arrays
    # dummy array
    rhsvpavperp = fkpl_arrays.fprp_solver_data.matrix_operators.rhsvpavperp
    # storage for Rosenbluth potentials
    rosenbluth_potentials_s = fkpl_arrays.rosenbluth_potentials_s
    # for each species, get the Rosenbluth potential due to that species
    if use_Maxwellian_Rosenbluth_coefficients
        for is in 1:species.n
            @views calculate_rosenbluth_potentials_via_analytical_Maxwellian!(rosenbluth_potentials_s[is],
                    ff_in[:,:,is],vpa,vperp,species.mass[is])
        end
    else
        for is in 1:species.n
            @views calculate_rosenbluth_potentials_via_elliptic_solve!(rosenbluth_potentials_s[is],ff_in[:,:,is],
                    vpa,vperp,fkpl_arrays.fprp_solver_data,species.mass[is],
                    algebraic_solve_for_d2Gdvperp2=algebraic_solve_for_d2Gdvperp2,
                    calculate_GG=calculate_GG,calculate_dGdvperp=calculate_dGdvperp)
        end
    end
    if fkpl_arrays.multi_species_operator_option == single_assembly_per_species
        # Version of the collision operator where we can assemble the collision operator
        # only once per species to reduce cost, but still get the correction terms for each
        # species species' pair of collison operators using finite-element integrals
        # storage for summed potentials
        rosenbluth_potentials = fkpl_arrays.rosenbluth_potentials
        fixed_background_plasma = fkpl_arrays.fixed_background_plasma
        # for each species, sum up the Rosenbluth potentials to make the appropriate
        # total Rosenbluth potential, and assemble the collision operator
        for is in 1:species.n
            calculate_cross_species_rosenbluth_potential_sums!(rosenbluth_potentials,
                    rosenbluth_potentials_s,species,species.zeds[is],species.mass[is],
                    fixed_background_plasma)
            # assemble weak form and solve mass matrix problem for CCssp
            @views fokker_planck_collision_operator_solve!(
                         CCs[:,:,is], ff_in[:,:,is], rosenbluth_potentials, 1.0, 1.0, nuref,
                         rhsvpavperp, lu_obj_MM, YY_arrays, vpa, vperp)
        end
    else
        # dummy array, unused as Rosenbluth potentials are already determined
        Cssp = fkpl_arrays.fprp_solver_data.matrix_operators.S_dummy
        mass = species.mass
        zeds = species.zeds
        # moments of collisions for each cross-species pair
        delta_n_sp_s = fkpl_arrays.delta_n_sp_s
        delta_m_sp_s = fkpl_arrays.delta_m_sp_s
        delta_p_sp_s = fkpl_arrays.delta_p_sp_s
        # moments of the pdf for each species needed for below calculation
        density = fkpl_arrays.density
        upar = fkpl_arrays.upar
        # collect the calculated moments
        for is in 1:species.n
            @views density[is] = get_density(ff_in[:,:,is], vpa, vperp)
            @views upar[is] = get_upar(ff_in[:,:,is], vpa, vperp, density[is])
        end
        @. CCs[:,:,:] = 0.0
        for is in 1:species.n
            for isp in 1:species.n
                nussp = nuref*(zeds[is]*zeds[isp]/mass[is])^2
                # assemble weak form and solve mass matrix problem for Cssp
                @views fokker_planck_collision_operator_solve!(
                            Cssp, ff_in[:,:,is], rosenbluth_potentials_s[isp], mass[is], mass[isp], nussp,
                            rhsvpavperp, lu_obj_MM, YY_arrays, vpa, vperp)
                # impose any non-natural boundary conditions
                enforce_vpavperp_BCs!(Cssp,vpa,vperp)
                # get moments
                delta_n_sp_s[isp,is] = get_density(Cssp, vpa, vperp)
                delta_m_sp_s[isp,is] = mass[is]*(get_upar(Cssp, vpa, vperp, 1.0) - upar[is]*delta_n_sp_s[isp,is])
                delta_p_sp_s[isp,is] = get_pressure(Cssp, vpa, vperp, upar[is], mass[is])
                # sum up the collision operator contributions
                @. CCs[:,:,is] += Cssp
            end
            # cross-species contributions from fixed background
            @views fokker_planck_cross_species_collision_operator!(
                        Cssp,
                        ff_in[:,:,is],
                        nuref, mass[is], zeds[is],
                        fkpl_arrays.rosenbluth_potentials,
                        fkpl_arrays.fixed_background_plasma,
                        rhsvpavperp, lu_obj_MM, YY_arrays, vpa, vperp;
                        use_conserving_corrections=use_conserving_corrections)
            @views CCs[:,:,is] += Cssp
        end
    end
    if use_conserving_corrections
        # apply multi-species conserving terms
        conserving_corrections!(CCs, ff_in, nuref, fkpl_arrays)
    end
    return nothing
end

"""
Cross-species collisions due to fixed background plasma.
"""
function fokker_planck_cross_species_collision_operator!(
                        CC::AbstractArray{mk_float,2},
                        ff_in::AbstractArray{mk_float,2},
                        nuref::mk_float, ms::mk_float, Zs::mk_float,
                        rosenbluth_potentials,
                        fixed_background_plasma,
                        rhsvpavperp, lu_obj_MM, YY_arrays, vpa, vperp;
                        use_conserving_corrections=true::Bool)
    # calculate the Rosenbluth potentials due to the background plasma
    calculate_cross_species_rosenbluth_potential_sums!(rosenbluth_potentials,
                    Zs,ms,fixed_background_plasma)
    # assemble weak form, solve mass matrix for CC at collocation points
    fokker_planck_collision_operator_solve!(
                    CC, ff_in, rosenbluth_potentials, 1.0, 1.0, nuref,
                    rhsvpavperp, lu_obj_MM, YY_arrays, vpa, vperp)
    if use_conserving_corrections
        # enforce the boundary conditions on CC before it is used for timestepping
        enforce_vpavperp_BCs!(CC,vpa,vperp)
        # make ad-hoc conserving corrections
        density_conserving_correction!(CC,ff_in,vpa,vperp)
    end
    return nothing
end

"""
Function to calculate entropy production.
"""
function calculate_entropy_production(CC::AbstractArray{mk_float,2},
                    pdf::AbstractArray{mk_float,2},
                    fkpl_arrays::fokkerplanck_weakform_arrays_struct)
    vpa = fkpl_arrays.vpa
    vperp = fkpl_arrays.vperp
    # assign dummy array
    lnfC = fkpl_arrays.fprp_solver_data.matrix_operators.rhsvpavperp
    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                lnfC[ivpa,ivperp] = log(abs(pdf[ivpa,ivperp]) + 1.0e-15)*CC[ivpa,ivperp]
            end
        end
    end
    dSdt = -get_density(lnfC,vpa,vperp)
    return dSdt
end
function calculate_entropy_production(
                    CCs::AbstractArray{mk_float,3},
                    pdf::AbstractArray{mk_float,3},
                    fkpl_arrays::fokkerplanck_weakform_arrays_struct)
    vpa = fkpl_arrays.vpa
    vperp = fkpl_arrays.vperp
    species = fkpl_arrays.species
    # assign dummy array
    lnfC = fkpl_arrays.fprp_solver_data.matrix_operators.rhsvpavperp
    dSdt = 0.0
    @inbounds begin
        # compute entropy production for each species,
        # and sum to get the total entropy production
        for is in 1:species.n
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    lnfC[ivpa,ivperp] = log(abs(pdf[ivpa,ivperp,is]) + 1.0e-15)*CCs[ivpa,ivperp,is]
                end
            end
            dSdt += -get_density(lnfC,vpa,vperp)
        end
    end
    return dSdt
end

######################################################
# end functions associated with the weak-form operator
# where the potentials are computed by elliptic solve
######################################################

#################################################
# Functions associated with implicit timestepping
#################################################

function fokker_planck_collisions_backward_euler_step!(Fold::AbstractArray{mk_float,3},
                        delta_t::mk_float, nuref::mk_float,
                        fkpl_arrays::fokker_planck_backward_euler_data;
                        use_conserving_corrections=true::Bool,
                        use_conserving_corrections_on_C=true::Bool,
                        test_linearised_advance=false::Bool,
                        test_particle_preconditioner=true::Bool,
                        use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=false::Bool,
                        update_test_particle_preconditioner=true::Bool)
    CCs = fkpl_arrays.CCs
    source = fkpl_arrays.source
    species = fkpl_arrays.fp_operator.species
    vperp = fkpl_arrays.fp_operator.vperp
    vpa = fkpl_arrays.fp_operator.vpa
    YY_arrays = fkpl_arrays.fp_operator.YY_arrays
    source_data = fkpl_arrays.source_data
    # residual function to be used for Newton-Krylov
    # residual(vpa, vperp, species) = F^(n+1) - F^n - dt * C[F^n+1,F^n+1]
    function residual_func!(Fresidual, Fnew; krylov=false)
        fokker_planck_collision_operator_weak_form!(CCs,
                        Fnew, nuref,
                        fkpl_arrays.fp_operator;
                        use_conserving_corrections=(use_conserving_corrections && use_conserving_corrections_on_C))
        @. source = 0.0 # set source to zero initially, in case source_data=nothing
        slowing_down_source!(source, fkpl_arrays.fp_operator, source_data)
        slowing_down_sink!(source, Fnew, fkpl_arrays.fp_operator, source_data)
        # N.B. residual function below is defined to be Residual =  dF/dt - RHS
        # evaluated for data at the collocation points, i.e., not in weak form.
        @inbounds begin
            for is in 1:species.n
                for ivperp in 1:vperp.n
                    for ivpa in 1:vpa.n
                        Fresidual[ivpa,ivperp,is] = Fnew[ivpa,ivperp,is] - Fold[ivpa,ivperp,is] - delta_t * (CCs[ivpa,ivperp,is] + source[ivpa,ivperp,is])
                    end
                end
            end
        end
        return nothing
    end

    if test_particle_preconditioner
        # test particle preconditioner CC2D_sparse is the matrix
        # K_ijkl = int phi_i(vpa)phi_j(vperp) ( phi_k(vpa)phi_l(vperp) - dt C[ phi_k(vpa)phi_l(vperp) , F^n(vpa,vperp) ])  vperp d vperp d vpa,
        # such that K * F^n+1 = M * F^n advances the linearised collision operator due
        # to test particle collisions only (differential piece of C).
        # CC2D_sparse is the approximate Jacobian for the residual Fresidual.
        if update_test_particle_preconditioner
            calculate_test_particle_preconditioner!(Fold,delta_t,nuref,fkpl_arrays,
                    use_Maxwellian_Rosenbluth_coefficients=use_Maxwellian_Rosenbluth_coefficients_in_preconditioner)
        end
        function test_particle_precon!(x)
            # let K * dF = C[dF,F^n]
            # function to solve K * F^n+1 = M * F^n
            # and return F^n+1 in place in x
            pdf = x
            advance_linearised_test_particle_collisions!(pdf,fkpl_arrays)
            return nothing
        end
        right_preconditioner = test_particle_precon!
    else
        right_preconditioner = nothing
    end
    # initial condition for Fnew for JFNK or linearised advance below
    Fnew = fkpl_arrays.Fs_new
    @inbounds begin
        for is in 1:species.n
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    Fnew[ivpa,ivperp,is] = Fold[ivpa,ivperp,is]
                end
            end
        end
    end
    if test_linearised_advance
        add_slowing_down_source!(Fnew, fkpl_arrays.Fsw,
            fkpl_arrays.fp_operator, source_data, delta_t)
        test_particle_precon!(Fnew)
        success = true
    else
        nl_solver_params = fkpl_arrays.nl_solver_data_s
        Fresidual = fkpl_arrays.Fs_residual
        F_delta_x = fkpl_arrays.Fs_delta_x
        F_rhs_delta = fkpl_arrays.Fs_rhs_delta
        Fv = fkpl_arrays.Fsv
        Fw = fkpl_arrays.Fsw
        success = newton_solve!(Fnew, residual_func!,
                        Fresidual, F_delta_x, F_rhs_delta, Fv, Fw, nl_solver_params;
                        right_preconditioner=right_preconditioner)
        # apply BCs on result, if non-natural BCs are imposed
        for is in 1:species.n
            @views enforce_vpavperp_BCs!(Fnew[:,:,is],vpa,vperp)
        end
        # should only introduce error of order ~ atol
        # so long as the system is closed, i.e., no sources and sinks or fixed species
        if use_conserving_corrections
            # ad-hoc end-of-step corrections, again introducing only ~atol error
            # correct Fnew = F^n+1 - F^n so it has no change in moments n,
            # and no change in the total momentum P = sum_s m_s n_s u_s, and total
            # energy E =  sum_s (3/2) p_s + (1/2) m_s n_s u_s^2
            # this introduces errors of the size of the distance between F^n+1 and the
            # "correct" root that should have been found by the iterative solve, i.e.,
            # errors of size ~ atol.
            conserving_corrections!(Fnew, Fold, fkpl_arrays.fp_operator, source_data)
        end
    end
    return success
end

end
