using Dates
using FokkerPlanck: FokkerPlanckBackwardEulerData,
                    fokker_planck_collisions_backward_euler_step!,
                    fokker_planck_collision_operator_weak_form!,
                    FokkerPlanckWeakformArrays,
                    multipole_expansion, delta_f_multipole, boundary_data_type,
                    multi_species_operator_type, single_assembly_per_species, repeat_assembly_per_species
# enum for controlling return data from test functions
@enum CollisionTestReturnType begin
    continuous_integration_test # if ci test, return initial and final pdfs for regression testing
    timing_test # return timing data
    interactive_test # return nothing
end
# provides functions for test below to keep this script concise
include(joinpath(@__DIR__,"ImplicitCollisionsTestBase.jl"))
function test_implicit_collisions(;
    # initial pdf info
    vth0::Float64=0.5, vperp0::Float64=1.0, vpa0::Float64=0.0, zbeam::Float64=0.0, nspecies::Int64=1,
    # grid info
    ngrid::Int64=3, nelement_vpa::Int64=8, nelement_vperp::Int64=4,
    Lvpa::Float64=6.0, Lvperp::Float64=3.0,
    # boundary condition info
    bc_vpa::Tbc_vpa=natural_boundary_condition,
    bc_vperp::Tbc_vperp=natural_boundary_condition,
    # time advance info
    ntime::Int64=1,delta_t::Float64=1.0,
    # nonlinea r solver options
    atol::Float64 = 1.0e-10, rtol::Float64 = 0.0,
    nonlinear_max_iterations::Int64 = 20, test_particle_preconditioner::Bool=true,
    # model options
    test_linearised_advance::Bool=false,
    use_Maxwellian_Rosenbluth_coefficients_in_preconditioner::Bool=false,
    test_numerical_conserving_terms::Bool=false,
    boundary_data_option::boundary_data_type=multipole_expansion,
    multi_species_operator_option::multi_species_operator_type=repeat_assembly_per_species,
    test_external_chebyshev_grid::Bool=false,
    print_diagnostics::Bool=true, print_timing::Bool=true,
    test_type::CollisionTestReturnType=interactive_test,
    # if external user, may pass a slice of an array into functions
    test_input_array_type::Bool=false
    ) where {Tbc_vpa <: AbstractBoundaryCondition, Tbc_vperp <: AbstractBoundaryCondition}
    vth0_range = [vth0 for is in 1:nspecies]
    vperp0_range = [vperp0 for is in 1:nspecies]
    vpa0_range = [vpa0 for is in 1:nspecies]
    zbeam_range = [zbeam for is in 1:nspecies]
    mass_range = [1.0 for is in 1:nspecies]
    zeds_range = [1.0 for is in 1:nspecies]
    density_range = [1.0/nspecies for is in 1:nspecies]
    c0ref_range = [1.0 for is in 1:nspecies]
    u0ref_range = [0.0 for is in 1:nspecies]
    n0ref_range = [1.0 for is in 1:nspecies]
    return test_multispecies_implicit_collisions(;
        # initial pdf info
        vth0=vth0_range, vperp0=vperp0_range, vpa0=vpa0_range, zbeam=zbeam_range,
        # species info
        mass=mass_range, zeds=zeds_range, c0ref=c0ref_range, u0ref=u0ref_range, n0ref=n0ref_range,
        density_in=density_range,
        # grid info
        ngrid=ngrid, nelement_vpa=nelement_vpa, nelement_vperp=nelement_vperp,
        Lvpa=Lvpa, Lvperp=Lvperp,
        # boundary condition info
        bc_vpa=bc_vpa,
        bc_vperp=bc_vperp,
        # time advance info
        ntime=ntime,delta_t=delta_t,
        # nonlinear solver options
        atol = atol, rtol = rtol,
        nonlinear_max_iterations = nonlinear_max_iterations, test_particle_preconditioner=test_particle_preconditioner,
        # model options
        test_linearised_advance=test_linearised_advance,
        use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=use_Maxwellian_Rosenbluth_coefficients_in_preconditioner,
        test_numerical_conserving_terms=test_numerical_conserving_terms,
        test_numerical_conserving_terms_on_C=test_numerical_conserving_terms,
        boundary_data_option=boundary_data_option,
        multi_species_operator_option=multi_species_operator_option,
        test_external_chebyshev_grid=test_external_chebyshev_grid,
        print_diagnostics=print_diagnostics, print_timing=print_timing,
        test_type=test_type,
        # if external user, may pass a slice of an array into functions
        test_input_array_type=test_input_array_type)
end

function test_multispecies_implicit_collisions(;
    # initial pdf info
    vth0::Vector{Float64}=[0.5,0.5], vperp0::Vector{Float64}=[1.0,1.0], vpa0::Vector{Float64}=[0.0,0.0], zbeam::Vector{Float64}=[0.0,0.0],
    # species info
    mass::Vector{Float64}=[1.0,1.0], zeds::Vector{Float64}=[1.0,1.0], c0ref::Vector{Float64}=[1.0,1.0],
    u0ref::Vector{Float64}=[0.0,0.0], n0ref::Vector{Float64}=[1.0,1.0], density_in::Vector{Float64}=[1.0,1.0],
    # grid info
    ngrid::Int64=3, nelement_vpa::Int64=8, nelement_vperp::Int64=4,
    Lvpa::Float64=6.0, Lvperp::Float64=3.0,
    # boundary condition info
    bc_vpa::Tbc_vpa=natural_boundary_condition,
    bc_vperp::Tbc_vperp=natural_boundary_condition,
    # time advance info
    ntime::Int64=1,delta_t::Float64=1.0,
    # nonlinear solver options
    atol::Float64 = 1.0e-10, rtol::Float64 = 0.0,
    nonlinear_max_iterations::Int64 = 20, test_particle_preconditioner::Bool=true,
    # model options
    test_linearised_advance::Bool=false,
    use_Maxwellian_Rosenbluth_coefficients_in_preconditioner::Bool=false,
    test_numerical_conserving_terms::Bool=false,
    test_numerical_conserving_terms_on_C::Bool=true,
    boundary_data_option::boundary_data_type=multipole_expansion,
    multi_species_operator_option::multi_species_operator_type=single_assembly_per_species,
    test_external_chebyshev_grid::Bool=false,
    print_diagnostics::Bool=true, print_timing::Bool=true,
    test_type::CollisionTestReturnType=interactive_test,
    # if external user, may pass a slice of an array into functions
    test_input_array_type::Bool=false
    ) where {Tbc_vpa <: AbstractBoundaryCondition, Tbc_vperp <: AbstractBoundaryCondition}

    # number of species
    nspecies = length(zeds)
    # check inputs are consistent
    @boundscheck (nspecies == length(mass) && nspecies == length(vth0)
                   && nspecies == length(vperp0) && nspecies == length(vpa0)
                   && nspecies == length(zbeam) && nspecies == length(density_in)
                   && nspecies == length(c0ref) && nspecies == length(u0ref)
                   && nspecies == length(n0ref)) || throw(BoundsError(zeds))
    # check density_in > 0
    for is in 1:nspecies
        if density_in[is] < 1.0e-14
            error("Require values of density_in to be greater than 0.999e-14")
        end
    end
    start_init_time = now()
    # group integer inputs using `ScalarCoordinateInputs` from FokkerPlanck.coordinates
    input_vpa_scalar = ScalarCoordinateInputs(ngrid, nelement_vpa, -0.5*Lvpa, 0.5*Lvpa, include_boundary_points)
    input_vperp_scalar = ScalarCoordinateInputs(ngrid, nelement_vperp, 0.0, Lvperp, exclude_lower_boundary_point)
    if test_external_chebyshev_grid
        # construct an instance of Array{ElementCoordinates,1} to use user-provided custom grid
        input_vpa = chebyshev_grid("vpa",input_vpa_scalar)
        input_vperp = chebyshev_grid("vperp",input_vperp_scalar)
    else
        # use the Gauss-Legendre grid constructed internally in FokkerPlanck.coordinates
        input_vpa = input_vpa_scalar
        input_vperp = input_vperp_scalar
    end
    # initialise all arrays needed to evaluate the nonlinear Fokker-Planck operator
    fkpl_arrays = FokkerPlanckBackwardEulerData(
                        mass, zeds, c0ref, u0ref, n0ref,
                        input_vpa,
                        input_vperp;
                        bc_vpa=bc_vpa,
                        bc_vperp=bc_vperp,
                        boundary_data_option=boundary_data_option,
                        multi_species_operator_option=multi_species_operator_option,
                        nl_solver_atol=atol,
                        nl_solver_rtol=rtol,
                        nl_solver_nonlinear_max_iterations=nonlinear_max_iterations,
                        print_to_screen=print_diagnostics)
    # extract coordinates
    vpa = fkpl_arrays.fp_operator.vpa
    vperp = fkpl_arrays.fp_operator.vperp
    species = fkpl_arrays.fp_operator.species
    # arrays needed for advance
    if test_input_array_type
        Fgeneral = Array{Float64}(undef,vpa.n,vperp.n,1,1,species.n)
        @views Fold = Fgeneral[:,:,1,1,:]
    else
        Fold = Array{Float64}(undef,vpa.n,vperp.n,species.n)
    end
    CC = fkpl_arrays.CCs # needed for dSdt diagnostic
    # dummy arrays needed for diagnostics
    Fout = Array{Float64}(undef,vpa.n,vperp.n,species.n,2)
    Fdummy1 = Array{Float64}(undef,vpa.n,vperp.n,species.n)
    Fdummy2 = Array{Float64}(undef,vpa.n,vperp.n)
    Fdummy3 = Array{Float64}(undef,vpa.n,vperp.n)
    moments = moments_struct(species.n)
    # physics parameters
    nuss = 1.0
    # initial condition
    time = 0.0
    for is in 1:species.n
        @views set_initial_pdf!(Fold[:,:,is],vpa,vperp,vpa0[is],vperp0[is],vth0[is],zbeam[is])
        # multiply by initial density
        @views Fold[:,:,is] *= density_in[is]
    end
    # store initial pdf for output
    Fout[:,:,:,1] .= Fold
    # get initial C[F,F] for entropy production diagnostic
    fokker_planck_collision_operator_weak_form!(CC, Fold, nuss, fkpl_arrays.fp_operator,
            use_conserving_corrections=test_numerical_conserving_terms_on_C)
    # print diagnostic info to screen, calculate moments
    diagnose_F_Maxwellian(CC, Fold,Fdummy1,Fdummy2,Fdummy3,fkpl_arrays.fp_operator,moments,time,0,print_diagnostics=print_diagnostics)
    finish_init_time = now()
    # time advance with backward Euler
    for it in 1:ntime
        # use Fold = F^n to obtain Fnew = F^n+1 for n = it
        fokker_planck_collisions_backward_euler_step!(Fold, delta_t, nuss, fkpl_arrays,
            use_conserving_corrections=test_numerical_conserving_terms,
            use_conserving_corrections_on_C=test_numerical_conserving_terms_on_C,
            test_particle_preconditioner=test_particle_preconditioner,
            test_linearised_advance=test_linearised_advance,
            use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=use_Maxwellian_Rosenbluth_coefficients_in_preconditioner)
        # update the pdf by extracting Fs_new from fkpl_arrays
        Fnew = fkpl_arrays.Fs_new
        @inbounds begin
            for is in 1:species.n
                for ivperp in 1:vperp.n
                    for ivpa in 1:vpa.n
                        Fold[ivpa,ivperp,is] = Fnew[ivpa,ivperp,is]
                    end
                end
            end
        end
        # diagnose the updated Fold
        time += delta_t
        diagnose_F_Maxwellian(CC,Fold,Fdummy1,Fdummy2,Fdummy3,fkpl_arrays.fp_operator,moments,time,it,print_diagnostics=print_diagnostics)
    end
    finish_run_time = now()
    # store final pdf for output
    Fout[:,:,:,2] .= Fold
    # println("total newton iterations: ", fkpl_arrays.nl_solver_data.nonlinear_iterations[])
    init_time = Dates.value(finish_init_time - start_init_time)
    run_time = Dates.value(finish_run_time - finish_init_time)
    if print_timing
        # print some timing information
        println("init time (ms): ", init_time)
        println("run time (ms): ", run_time)
    end
    # to make this function testable
    if test_type == continuous_integration_test
        # uncomment to update tests
        # print_grid(vpa)
        # print_grid(vperp)
        # print_pdf(Fout)
        return pdf_and_grid(vpa.grid,vperp.grid,Fout,moments)
    elseif test_type == timing_test
        return init_time, run_time
    else # interactive_test
        return nothing
    end
end

# provides functions for slowing down calculations
include(joinpath(@__DIR__,"SlowingDownTestBase.jl"))
function test_implicit_slowing_down(;
    # initial pdf info
    vth0::Vector{Float64}=[0.5], vperp0::Vector{Float64}=[1.0], vpa0::Vector{Float64}=[0.0], zbeam::Vector{Float64}=[0.0],
    # source info
    source_rate::Vector{Float64}=[1.0], source_v0::Vector{Float64}=[1.0], source_vth::Vector{Float64}=[0.05], sink_rate::Vector{Float64}=[5.0], sink_vth::Float64=0.05, constant_sink::Bool=false,
    # species info
    mass::Vector{Float64}=[1.0], zeds::Vector{Float64}=[2.0], c0ref::Vector{Float64}=[1.0],
    u0ref::Vector{Float64}=[0.0], n0ref::Vector{Float64}=[1.0], density_in::Vector{Float64}=[1.0e-8],
    # grid info
    ngrid::Int64=5, nelement_vpa::Int64=32, nelement_vperp::Int64=16,
    Lvpa::Float64=3.0, Lvperp::Float64=1.5,
    # boundary condition info
    bc_vpa::Tbc_vpa=natural_boundary_condition,
    bc_vperp::Tbc_vperp=natural_boundary_condition,
    # time advance info
    ntime::Int64=1,delta_t::Float64=0.01,
    # nonlinear solver options
    atol::Float64 = 1.0e-10, rtol::Float64 = 0.0,
    nonlinear_max_iterations::Int64 = 20, test_particle_preconditioner::Bool=true,
    # model options
    electron_mass::Float64 = 1.0/1836.0, thermal_temperature::Float64 = 0.01,
    test_linearised_advance::Bool=false,
    use_Maxwellian_Rosenbluth_coefficients_in_preconditioner::Bool=false,
    test_numerical_conserving_terms::Bool=true,
    test_numerical_conserving_terms_on_C::Bool=true,
    boundary_data_option::boundary_data_type=multipole_expansion,
    multi_species_operator_option::multi_species_operator_type=single_assembly_per_species,
    print_diagnostics::Bool=true, print_timing::Bool=true, print_final_pdf::Bool=false
    ) where {Tbc_vpa <: AbstractBoundaryCondition, Tbc_vperp <: AbstractBoundaryCondition}

    # number of species
    nspecies = length(zeds)
    # check inputs are consistent
    @boundscheck (nspecies == length(mass) && nspecies == length(vth0)
                   && nspecies == length(vperp0) && nspecies == length(vpa0)
                   && nspecies == length(zbeam) && nspecies == length(density_in)
                   && nspecies == length(c0ref) && nspecies == length(u0ref)
                   && nspecies == length(n0ref)) || throw(BoundsError(zeds))
    # check density_in > 0
    for is in 1:nspecies
        if density_in[is] < 1.0e-14
            error("Require values of density_in to be greater than 0.999e-14")
        end
    end
    start_init_time = now()
    # group integer inputs using `ScalarCoordinateInputs` from FokkerPlanck.coordinates
    input_vpa = ScalarCoordinateInputs(ngrid, nelement_vpa, -0.5*Lvpa, 0.5*Lvpa, include_boundary_points)
    input_vperp = ScalarCoordinateInputs(ngrid, nelement_vperp, 0.0, Lvperp, exclude_lower_boundary_point)
    # fixed background Maxwellian inputs
    # parameters of fixed background species
    msp = [0.25*electron_mass, 0.25]
    Zsp = [-1.0, 1.0]
    denssp = [1.0,1.0]
    uparsp = [0.0,0.0]
    temp = thermal_temperature
    vthsp = [sqrt(2.0*temp/msp[1]), sqrt(2.0*temp/msp[2])]
    fixed_background_plasma_in = FixedBackgroundPlasmaInput(msp,Zsp,denssp,uparsp,vthsp)
    # information for sources of the evolving species
    source_data_in = SlowingDownSourceInput(source_rate,source_vth,source_v0,sink_rate,sink_vth,constant_sink)
    # initialise all arrays needed to evaluate the nonlinear Fokker-Planck operator
    fkpl_arrays = FokkerPlanckBackwardEulerData(
                        mass, zeds, c0ref, u0ref, n0ref,
                        input_vpa,
                        input_vperp;
                        bc_vpa=bc_vpa,
                        bc_vperp=bc_vperp,
                        boundary_data_option=boundary_data_option,
                        multi_species_operator_option=multi_species_operator_option,
                        nl_solver_atol=atol,
                        nl_solver_rtol=rtol,
                        nl_solver_nonlinear_max_iterations=nonlinear_max_iterations,
                        print_to_screen=print_diagnostics,
                        fixed_background_plasma_in=fixed_background_plasma_in,
                        source_data_in=source_data_in)
    # extract coordinates
    vpa = fkpl_arrays.fp_operator.vpa
    vperp = fkpl_arrays.fp_operator.vperp
    species = fkpl_arrays.fp_operator.species
    # arrays needed for advance
    Fold = Array{Float64}(undef,vpa.n,vperp.n,species.n)
    CC = fkpl_arrays.CCs # needed for dSdt diagnostic
    # dummy arrays needed for diagnostics
    Fout = Array{Float64}(undef,vpa.n,vperp.n,species.n,2)
    Fdummy1 = Array{Float64}(undef,vpa.n,vperp.n,species.n)
    Fdummy2 = Array{Float64}(undef,vpa.n,vperp.n)
    Fdummy3 = Array{Float64}(undef,vpa.n,vperp.n)
    moments = moments_struct(species.n)
    # physics parameters
    nuref = 1.0
    # initial condition
    time = 0.0
    for is in 1:species.n
        @views set_initial_pdf!(Fold[:,:,is],vpa,vperp,vpa0[is],vperp0[is],vth0[is],zbeam[is])
        # multiply by initial density
        @views Fold[:,:,is] *= density_in[is]
    end
    # store initial pdf for output
    Fout[:,:,:,1] .= Fold
    # get initial C[F,F] for entropy production diagnostic
    fokker_planck_collision_operator_weak_form!(CC, Fold, nuref, fkpl_arrays.fp_operator,
            use_conserving_corrections=test_numerical_conserving_terms_on_C)
    # analytical slowing down pdf for diagnostics
    sd_pdf = slowing_down_pdf(vpa, vperp, species, source_rate, source_v0, source_vth,
        msp, Zsp, denssp, vthsp, nuref)
    sd_errors = SD_error_data(species.n)
    # print diagnostic info to screen, calculate moments
    diagnose_F_Maxwellian(CC, Fold,Fdummy1,Fdummy2,Fdummy3,fkpl_arrays.fp_operator,moments,time,0,print_diagnostics=print_diagnostics)
    diagnose_F_SD!(sd_errors, sd_pdf, Fold, Fdummy1, Fdummy2, Fdummy3,
        source_v0, vthsp[2], vpa, vperp, species,
        print_to_screen=print_diagnostics)
    finish_init_time = now()
    # time advance with backward Euler
    for it in 1:ntime
        if test_linearised_advance
            # do not update the preconditioner (used as the time-advance matrix) after initialisation
            update_test_particle_preconditioner = it==1
        else
            update_test_particle_preconditioner = true
        end
        # use Fold = F^n to obtain Fnew = F^n+1 for n = it
        fokker_planck_collisions_backward_euler_step!(Fold, delta_t, nuref, fkpl_arrays,
            use_conserving_corrections=test_numerical_conserving_terms,
            use_conserving_corrections_on_C=test_numerical_conserving_terms_on_C,
            test_particle_preconditioner=test_particle_preconditioner,
            test_linearised_advance=test_linearised_advance,
            use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=use_Maxwellian_Rosenbluth_coefficients_in_preconditioner,
            update_test_particle_preconditioner=update_test_particle_preconditioner)
        # update the pdf by extracting Fs_new from fkpl_arrays
        Fnew = fkpl_arrays.Fs_new
        @inbounds begin
            for is in 1:species.n
                for ivperp in 1:vperp.n
                    for ivpa in 1:vpa.n
                        Fold[ivpa,ivperp,is] = Fnew[ivpa,ivperp,is]
                    end
                end
            end
        end
        # diagnose the updated Fold
        time += delta_t
        diagnose_F_Maxwellian(CC,Fold,Fdummy1,Fdummy2,Fdummy3,fkpl_arrays.fp_operator,moments,time,it,updated_CC=!test_linearised_advance,print_diagnostics=print_diagnostics)
        diagnose_F_SD!(sd_errors, sd_pdf, Fold, Fdummy1, Fdummy2, Fdummy3,
            source_v0, vthsp[2], vpa, vperp, species,
            print_to_screen=print_diagnostics)
    end
    finish_run_time = now()
    # store final pdf for output
    Fout[:,:,:,2] .= Fold
    # println("total newton iterations: ", fkpl_arrays.nl_solver_data.nonlinear_iterations[])
    init_time = Dates.value(finish_init_time - start_init_time)
    run_time = Dates.value(finish_run_time - finish_init_time)
    if print_timing
        # print some timing information
        println("init time (ms): ", init_time)
        println("run time (ms): ", run_time)
    end
    # print_grid(vpa)
    # print_grid(vperp)
    # print_pdf(Fout)
    pdfandgrid = pdf_and_grid(vpa.grid,vperp.grid,Fout,moments)
    if print_final_pdf
        for is in 1:species.n
            v0 = source_v0[is]
            vth = source_vth[is]
            for ivperp in 1:vperp.n
                ivpamid = Int(floor(vpa.n/2) + 1)
                println(pdfandgrid.pdf[ivpamid,ivperp,is,2], " ", sd_pdf[ivpamid,ivperp,is], " ", sd_pdf[ivpamid,ivperp,is]/pdfandgrid.pdf[ivpamid,ivperp,is,2], " ", vperp.grid[ivperp], " ", shape_function_from_source(vperp.grid[ivperp],v0,vth)/shape_function_from_source(0.0,v0,vth)   )
            end
        end
    end
    return pdfandgrid, sd_pdf, sd_errors
end


if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")
    # run once to precompile
    test_implicit_collisions(test_particle_preconditioner=true,test_numerical_conserving_terms=true,
        vth0=0.5,vperp0=1.0,vpa0=1.0, nelement_vpa=4,nelement_vperp=2,Lvpa=8.0,Lvperp=4.0,
        bc_vpa=natural_boundary_condition, bc_vperp=natural_boundary_condition,
        ntime=1, delta_t = 1.0, ngrid=5, test_linearised_advance=false)
    # run a standard case now we are precompiled
    test_implicit_collisions(test_particle_preconditioner=true,test_numerical_conserving_terms=true,
        vth0=0.5,vperp0=1.0,vpa0=1.0, nelement_vpa=32,nelement_vperp=16,Lvpa=8.0,Lvperp=4.0,
        bc_vpa=natural_boundary_condition, bc_vperp=natural_boundary_condition,
        ntime=100, delta_t = 1.0, ngrid=5, test_linearised_advance=false)
end
