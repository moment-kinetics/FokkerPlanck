"""
Module for including the Full-F Fokker-Planck Collision Operator.

The functions in this module are split into two groups. 

The first set of functions implement the weak-form
Collision operator using the Rosenbluth-MacDonald-Judd
formulation in a divergence form. The Green's functions
for the Rosenbluth potentials are used to obtain the Rosenbluth
potentials at the boundaries. To find the potentials
everywhere else elliptic solves of the PDEs for the
Rosenbluth potentials are performed with Dirichlet
boundary conditions. These routines provide the default collision operator
used in the code.

The second set of functions are used to set up the necessary arrays to 
compute the Rosenbluth potentials everywhere in vpa, vperp
by direct integration of the Green's functions. These functions are 
supported for the purposes of testing and debugging.
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

export init_fokker_planck_collisions
# testing
export fokker_planck_collision_operator_weak_form!
export fokker_planck_self_collision_operator_weak_form!
export fokker_planck_cross_species_collision_operator_Maxwellian_Fsp!
export calculate_entropy_production
# implicit advance
export fokker_planck_self_collisions_backward_euler_step!

using Dates
using LinearAlgebra: lu, ldiv!
using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float
using ..coordinates: finite_element_coordinate, scalar_coordinate_inputs
using ..velocity_moments: get_density
using ..fokker_planck_calculus: fokkerplanck_weakform_arrays_struct,
                                assemble_explicit_collision_operator_rhs_serial!,
                                enforce_vpavperp_BCs!,
                                calculate_rosenbluth_potentials_via_elliptic_solve!,
                                calculate_rosenbluth_potentials_via_analytical_Maxwellian!,
                                calculate_test_particle_preconditioner!,
                                advance_linearised_test_particle_collisions!,
                                multipole_expansion, direct_integration, delta_f_multipole, boundary_data_type,
                                conserving_corrections!, density_conserving_correction!
using ..fokker_planck_test: d2Gdvpa2_Maxwellian, d2Gdvperpdvpa_Maxwellian, d2Gdvperp2_Maxwellian, dHdvpa_Maxwellian, dHdvperp_Maxwellian,
                            F_Maxwellian, dFdvpa_Maxwellian, dFdvperp_Maxwellian
using JacobianFreeNewtonKrylov: newton_solve!
using FiniteElementMatrices: element_coordinates

"""
Wrapper function to provide the interface for initialising the
Fokker Planck operator arrays and operators. We require that
the inputs are provided with the types
```
    inputs =  scalar_coordinate_inputs(ngrid, nelement, L)
```
or 
```
    inputs = Array{element_coordinates,1}(undef, nelement)
```
where the former type is defined in `FokkerPlanck.coordinates`
and the latterr is defined in `FiniteElementMatrices`.
"""
function init_fokker_planck_collisions(
    inputs_vpa::Union{scalar_coordinate_inputs,Array{element_coordinates,1}},
    inputs_vperp::Union{scalar_coordinate_inputs,Array{element_coordinates,1}};
    bc_vpa="none"::String,
    bc_vperp="none"::String,
    boundary_data_option=multipole_expansion::boundary_data_type,
    nl_solver_atol=1.0e-10::mk_float,
    nl_solver_rtol=0.0::mk_float,
    nl_solver_nonlinear_max_iterations=20::mk_int,
    print_to_screen=true::Bool)

    # create the coordinate structs from the input data
    vperp = finite_element_coordinate("vperp", inputs_vperp,
                                bc=bc_vperp)
    vpa = finite_element_coordinate("vpa", inputs_vpa,
                                bc=bc_vpa)
    # use constructor function for fokkerplanck_weakform_arrays_struct
    return fokkerplanck_weakform_arrays_struct(vpa,vperp,boundary_data_option;
                nl_solver_atol=nl_solver_atol,
                nl_solver_rtol=nl_solver_rtol,
                nl_solver_nonlinear_max_iterations=nl_solver_nonlinear_max_iterations,
                    print_to_screen=print_to_screen)
end

function fokker_planck_self_collision_operator_weak_form!(
                         pdf_in::AbstractArray{mk_float,2}, ms::mk_float, nuss::mk_float,
                         fkpl_arrays::fokkerplanck_weakform_arrays_struct;
                         use_conserving_corrections=false::Bool)
    # first argument is Fs, and second argument is Fs' in C[Fs,Fs'] 
    @views fokker_planck_collision_operator_weak_form!(
        pdf_in, pdf_in, ms, ms, nuss, fkpl_arrays)
    CC = fkpl_arrays.CC
    vpa = fkpl_arrays.vpa
    vperp = fkpl_arrays.vperp
    # enforce the boundary conditions on CC before it is used for timestepping
    enforce_vpavperp_BCs!(CC,vpa,vperp)
    # make ad-hoc conserving corrections appropriate only for the self operator
    if use_conserving_corrections
        conserving_corrections!(CC, pdf_in, vpa, vperp)
    end
    return nothing
end

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
    rhsvpavperp = fkpl_arrays.rhsvpavperp
    lu_obj_MM = fkpl_arrays.lu_obj_MM
    YY_arrays = fkpl_arrays.YY_arrays    
    
    CC = fkpl_arrays.CC
    GG = fkpl_arrays.GG
    HH = fkpl_arrays.HH
    dHdvpa = fkpl_arrays.dHdvpa
    dHdvperp = fkpl_arrays.dHdvperp
    dGdvperp = fkpl_arrays.dGdvperp
    d2Gdvperp2 = fkpl_arrays.d2Gdvperp2
    d2Gdvpa2 = fkpl_arrays.d2Gdvpa2
    d2Gdvperpdvpa = fkpl_arrays.d2Gdvperpdvpa
    FF = fkpl_arrays.FF
    dFdvpa = fkpl_arrays.dFdvpa
    dFdvperp = fkpl_arrays.dFdvperp
    
    if use_Maxwellian_Rosenbluth_coefficients
        calculate_rosenbluth_potentials_via_analytical_Maxwellian!(GG,HH,dHdvpa,dHdvperp,
                 d2Gdvpa2,dGdvperp,d2Gdvperpdvpa,d2Gdvperp2,ffsp_in,vpa,vperp,msp)
    else
        calculate_rosenbluth_potentials_via_elliptic_solve!(GG,HH,dHdvpa,dHdvperp,
             d2Gdvpa2,dGdvperp,d2Gdvperpdvpa,d2Gdvperp2,ffsp_in,
             vpa,vperp,fkpl_arrays,
             algebraic_solve_for_d2Gdvperp2=algebraic_solve_for_d2Gdvperp2,
             calculate_GG=calculate_GG,calculate_dGdvperp=calculate_dGdvperp)
    end
    # assemble the RHS of the collision operator matrix eq
    assemble_explicit_collision_operator_rhs_serial!(rhsvpavperp,ffs_in,
          d2Gdvpa2,d2Gdvperpdvpa,d2Gdvperp2,
          dHdvpa,dHdvperp,ms,msp,nussp,
          vpa,vperp,YY_arrays)
    # solve the collision operator matrix eq
    # sc and rhsc are 1D views of the data in CC and rhsc, created so that we can use
    # the 'matrix solve' functionality of ldiv!() from the LinearAlgebra package
    sc = vec(CC)
    rhsc = vec(rhsvpavperp)
    # invert mass matrix and fill fc
    ldiv!(sc, lu_obj_MM, rhsc)
    return nothing
end

function fokker_planck_cross_species_collision_operator_Maxwellian_Fsp!(
                        ffs_in::AbstractArray{mk_float,2},
                        nuref::mk_float, ms::mk_float, Zs::mk_float,
                        msp::Array{mk_float,1}, Zsp::Array{mk_float,1},
                        densp::Array{mk_float,1}, uparsp::Array{mk_float,1},
                        vthsp::Array{mk_float,1},
                        fkpl_arrays::fokkerplanck_weakform_arrays_struct;
                        use_conserving_corrections=true::Bool)

    fokker_planck_collision_operator_weak_form_Maxwellian_Fsp!(ffs_in,
        nuref,ms,Zs,msp,Zsp,densp,uparsp,vthsp,
        fkpl_arrays)
    if use_conserving_corrections
        vpa = fkpl_arrays.vpa
        vperp = fkpl_arrays.vperp
        # enforce the boundary conditions on CC before it is used for timestepping
        enforce_vpavperp_BCs!(fkpl_arrays.CC,vpa,vperp)
        # make ad-hoc conserving corrections
        density_conserving_correction!(fkpl_arrays.CC,ffs_in,vpa,vperp)
    end
    return nothing
end

"""
Function for computing the collision operator
```math
\\sum_{s^\\prime} C[F_{s},F_{s^\\prime}]
```
when \$F_{s^\\prime}\$
is an analytically specified Maxwellian distribution and
the corresponding Rosenbluth potentials
are specified using analytical results.
"""
function fokker_planck_collision_operator_weak_form_Maxwellian_Fsp!(
                         ffs_in::AbstractArray{mk_float,2},
                         nuref::mk_float, ms::mk_float, Zs::mk_float,
                         msp::Array{mk_float,1}, Zsp::Array{mk_float,1},
                         densp::Array{mk_float,1}, uparsp::Array{mk_float,1},
                         vthsp::Array{mk_float,1},
                         fkpl_arrays::fokkerplanck_weakform_arrays_struct)
    vpa = fkpl_arrays.vpa
    vperp = fkpl_arrays.vperp
    @boundscheck vpa.n == size(ffs_in,1) || throw(BoundsError(ffs_in))
    @boundscheck vperp.n == size(ffs_in,2) || throw(BoundsError(ffs_in))
    
    # extract the necessary precalculated and buffer arrays from fokkerplanck_arrays
    rhsvpavperp = fkpl_arrays.rhsvpavperp
    lu_obj_MM = fkpl_arrays.lu_obj_MM
    YY_arrays = fkpl_arrays.YY_arrays    
    
    CC = fkpl_arrays.CC
    GG = fkpl_arrays.GG
    HH = fkpl_arrays.HH
    dHdvpa = fkpl_arrays.dHdvpa
    dHdvperp = fkpl_arrays.dHdvperp
    dGdvperp = fkpl_arrays.dGdvperp
    d2Gdvperp2 = fkpl_arrays.d2Gdvperp2
    d2Gdvpa2 = fkpl_arrays.d2Gdvpa2
    d2Gdvperpdvpa = fkpl_arrays.d2Gdvperpdvpa
    FF = fkpl_arrays.FF
    dFdvpa = fkpl_arrays.dFdvpa
    dFdvperp = fkpl_arrays.dFdvperp
    
    # number of primed species
    nsp = size(msp,1)
    
    # first set dummy arrays for coefficients to zero
    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                d2Gdvpa2[ivpa,ivperp] = 0.0
                d2Gdvperp2[ivpa,ivperp] = 0.0
                d2Gdvperpdvpa[ivpa,ivperp] = 0.0
                dHdvpa[ivpa,ivperp] = 0.0
                dHdvperp[ivpa,ivperp] = 0.0
            end
        end
    end
    # sum the contributions from the potentials, including order unity factors that differ between species
    # making use of the Linearity of the operator in Fsp
    # note that here we absorb ms/msp and Zsp^2 into the definition of the potentials, and we pass
    # ms = msp = 1 to the collision operator assembly routine so that we can use a single array to include
    # the contribution to the summed Rosenbluth potential from all the species
    for isp in 1:nsp
        dens = densp[isp]
        upar = uparsp[isp]
        vth = vthsp[isp]
        ZZ = (Zsp[isp]*Zs)^2 # factor from gamma_ss'
        @inbounds begin
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    d2Gdvpa2[ivpa,ivperp] += ZZ*d2Gdvpa2_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                    d2Gdvperp2[ivpa,ivperp] += ZZ*d2Gdvperp2_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                    d2Gdvperpdvpa[ivpa,ivperp] += ZZ*d2Gdvperpdvpa_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                    dHdvpa[ivpa,ivperp] += ZZ*(ms/msp[isp])*dHdvpa_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                    dHdvperp[ivpa,ivperp] += ZZ*(ms/msp[isp])*dHdvperp_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                end
            end
        end
    end
    # Need to synchronize as these arrays may be read outside the locally-owned set of
    # ivperp, ivpa indices in assemble_explicit_collision_operator_rhs_parallel!()
    # assemble the RHS of the collision operator matrix eq
    assemble_explicit_collision_operator_rhs_serial!(rhsvpavperp,ffs_in,
      d2Gdvpa2,d2Gdvperpdvpa,d2Gdvperp2,
      dHdvpa,dHdvperp,1.0,1.0,nuref,
      vpa,vperp,YY_arrays)

    # solve the collision operator matrix eq
    # sc and rhsc are 1D views of the data in CC and rhsc, created so that we can use
    # the 'matrix solve' functionality of ldiv!() from the LinearAlgebra package
    sc = vec(CC)
    rhsc = vec(rhsvpavperp)
    # invert mass matrix and fill fc
    ldiv!(sc, lu_obj_MM, rhsc)
    return nothing
end

"""
Function to calculate entropy production.
"""
function calculate_entropy_production(pdf::AbstractArray{mk_float,2},
                    fkpl_arrays::fokkerplanck_weakform_arrays_struct)
    # extract collision operator
    CC = fkpl_arrays.CC
    vpa = fkpl_arrays.vpa
    vperp = fkpl_arrays.vperp
    # assign dummy array
    lnfC = fkpl_arrays.rhsvpavperp
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


######################################################
# end functions associated with the weak-form operator
# where the potentials are computed by elliptic solve
######################################################

#################################################
# Functions associated with implicit timestepping
#################################################

function fokker_planck_self_collisions_backward_euler_step!(Fold::AbstractArray{mk_float,2},
                        delta_t::mk_float, ms::mk_float, nuss::mk_float,
                        fkpl_arrays::fokkerplanck_weakform_arrays_struct;
                        test_numerical_conserving_terms=false::Bool,
                        test_linearised_advance=false::Bool,
                        test_particle_preconditioner=true::Bool,
                        use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=false::Bool)

    vperp = fkpl_arrays.vperp
    vpa = fkpl_arrays.vpa

    # residual function to be used for Newton-Krylov
    # residual(vpa, vperp) = F^(n+1) - F^n - dt * C[F^n+1,F^n+1]
    function residual_func!(Fresidual, Fnew; krylov=false)
        fokker_planck_self_collision_operator_weak_form!(
                        Fnew, ms, nuss,
                        fkpl_arrays; 
                        use_conserving_corrections=test_numerical_conserving_terms)

        @inbounds begin
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    Fresidual[ivpa,ivperp] = Fnew[ivpa,ivperp] - Fold[ivpa,ivperp] - delta_t * (fkpl_arrays.CC[ivpa,ivperp])
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
        calculate_test_particle_preconditioner!(Fold,delta_t,ms,ms,nuss,fkpl_arrays,
                    use_Maxwellian_Rosenbluth_coefficients=use_Maxwellian_Rosenbluth_coefficients_in_preconditioner)
    
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
    Fnew = fkpl_arrays.Fnew
    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                Fnew[ivpa,ivperp] = Fold[ivpa,ivperp]
            end
        end
    end
    if test_linearised_advance
        test_particle_precon!(Fnew)
    else
        nl_solver_params = fkpl_arrays.nl_solver_data
        Fresidual = fkpl_arrays.Fresidual
        F_delta_x = fkpl_arrays.F_delta_x
        F_rhs_delta = fkpl_arrays.F_rhs_delta
        Fv = fkpl_arrays.Fv
        Fw = fkpl_arrays.Fw
        success = newton_solve!(Fnew, residual_func!,
                        Fresidual, F_delta_x, F_rhs_delta, Fv, Fw, nl_solver_params;
                        right_preconditioner=right_preconditioner)
        # apply BCs on result, if non-natural BCs are imposed
        # should only introduce error of order ~ atol
        enforce_vpavperp_BCs!(Fnew,vpa,vperp)
        if test_numerical_conserving_terms
            # ad-hoc end-of-step corrections, again introducing only ~atol error
            deltaF = fkpl_arrays.rhsvpavperp
            @inbounds begin
                for ivperp in 1:vperp.n
                    for ivpa in 1:vpa.n
                        deltaF[ivpa,ivperp] = Fnew[ivpa,ivperp] - Fold[ivpa,ivperp]
                    end
                end
            end
            # correct deltaF = F^n+1 - F^n so it has no change in moments n, u, p
            # this introduces errors of the size of the distance between F^n+1 and the 
            # "correct" root that should have been found by the iterative solve, i.e.,
            # errors of size ~ atol.
            conserving_corrections!(deltaF, Fold, vpa, vperp)
            # update Fnew
            @inbounds begin
                for ivperp in 1:vperp.n
                    for ivpa in 1:vpa.n
                        Fnew[ivpa,ivperp] = deltaF[ivpa,ivperp] + Fold[ivpa,ivperp]
                    end
                end
            end
        end
    end
    return success
end

end
