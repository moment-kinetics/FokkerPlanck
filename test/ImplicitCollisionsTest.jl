using Dates
using FokkerPlanck.array_allocation: allocate_float
using FokkerPlanck.type_definitions: mk_float, mk_int
using FokkerPlanck: fokker_planck_backward_euler_data,
                    fokker_planck_collisions_backward_euler_step!,
                    fokker_planck_collision_operator_weak_form!,
                    fokkerplanck_weakform_arrays_struct,
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
    vth0=0.5::mk_float, vperp0=1.0::mk_float, vpa0=0.0::mk_float, zbeam=0.0::mk_float, nspecies=1::mk_int,
    # grid info
    ngrid=3::mk_int, nelement_vpa=8::mk_int, nelement_vperp=4::mk_int,
    Lvpa=6.0::mk_float, Lvperp=3.0::mk_float,
    # boundary condition info
    bc_vpa=natural_boundary_condition::finite_element_boundary_condition_type,
    bc_vperp=natural_boundary_condition::finite_element_boundary_condition_type,
    # time advance info
    ntime=1::mk_int,delta_t=1.0::mk_float,
    # nonlinea r solver options
    atol = 1.0e-10::mk_float, rtol = 0.0::mk_float,
    nonlinear_max_iterations = 20::mk_int, test_particle_preconditioner=true::Bool,
    # model options
    test_linearised_advance=false::Bool,
    use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=false::Bool,
    test_numerical_conserving_terms=false::Bool,
    boundary_data_option=multipole_expansion::boundary_data_type,
    multi_species_operator_option=repeat_assembly_per_species::multi_species_operator_type,
    test_external_chebyshev_grid=false::Bool,
    print_diagnostics=true::Bool, print_timing=true::Bool,
    test_type=interactive_test::CollisionTestReturnType,
    # if external user, may pass a slice of an array into functions
    test_input_array_type=false::Bool)
    vth0_range = [vth0 for is in 1:nspecies]
    vperp0_range = [vperp0 for is in 1:nspecies]
    vpa0_range = [vpa0 for is in 1:nspecies]
    zbeam_range = [zbeam for is in 1:nspecies]
    mass_range = [1.0 for is in 1:nspecies]
    zeds_range = [1.0 for is in 1:nspecies]
    density_range = [1.0/nspecies for is in 1:nspecies]
    return test_multispecies_implicit_collisions(;
        # initial pdf info
        vth0=vth0_range, vperp0=vperp0_range, vpa0=vpa0_range, zbeam=zbeam_range,
        # species info
        mass=mass_range, zeds=zeds_range, density_in=density_range,
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
    vth0=[0.5,0.5]::Vector{mk_float}, vperp0=[1.0,1.0]::Vector{mk_float}, vpa0=[0.0,0.0]::Vector{mk_float}, zbeam=[0.0,0.0]::Vector{mk_float},
    # species info
    mass=[1.0,1.0]::Vector{mk_float}, zeds=[1.0,1.0]::Vector{mk_float}, density_in=[1.0,1.0]::Vector{mk_float},
    # grid info
    ngrid=3::mk_int, nelement_vpa=8::mk_int, nelement_vperp=4::mk_int,
    Lvpa=6.0::mk_float, Lvperp=3.0::mk_float,
    # boundary condition info
    bc_vpa=natural_boundary_condition::finite_element_boundary_condition_type,
    bc_vperp=natural_boundary_condition::finite_element_boundary_condition_type,
    # time advance info
    ntime=1::mk_int,delta_t=1.0::mk_float,
    # nonlinear solver options
    atol = 1.0e-10::mk_float, rtol = 0.0::mk_float,
    nonlinear_max_iterations = 20::mk_int, test_particle_preconditioner=true::Bool,
    # model options
    test_linearised_advance=false::Bool,
    use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=false::Bool,
    test_numerical_conserving_terms=false::Bool,
    test_numerical_conserving_terms_on_C=true::Bool,
    boundary_data_option=multipole_expansion::boundary_data_type,
    multi_species_operator_option=single_assembly_per_species::multi_species_operator_type,
    test_external_chebyshev_grid=false::Bool,
    print_diagnostics=true::Bool, print_timing=true::Bool,
    test_type=interactive_test::CollisionTestReturnType,
    # if external user, may pass a slice of an array into functions
    test_input_array_type=false::Bool)

    # number of species
    nspecies = length(zeds)
    # check inputs are consistent
    @boundscheck (nspecies == length(mass) && nspecies == length(vth0)
                   && nspecies == length(vperp0) && nspecies == length(vpa0)
                   && nspecies == length(zbeam) && nspecies == length(density_in)) || throw(BoundsError(zeds))
    # check density_in > 0
    for is in 1:nspecies
        if density_in[is] < 1.0e-14
            error("Require values of density_in to be greater than 0.999e-14")
        end
    end
    start_init_time = now()
    # group integer inputs using `scalar_coordinate_inputs` from FokkerPlanck.coordinates
    input_vpa_scalar = scalar_coordinate_inputs(ngrid, nelement_vpa, Lvpa)
    input_vperp_scalar = scalar_coordinate_inputs(ngrid, nelement_vperp, Lvperp)
    if test_external_chebyshev_grid
        # construct an instance of Array{element_coordinates,1} to use user-provided custom grid
        input_vpa = chebyshev_grid("vpa",input_vpa_scalar)
        input_vperp = chebyshev_grid("vperp",input_vperp_scalar)
    else
        # use the Gauss-Legendre grid constructed internally in FokkerPlanck.coordinates
        input_vpa = input_vpa_scalar
        input_vperp = input_vperp_scalar
    end
    # initialise all arrays needed to evaluate the nonlinear Fokker-Planck operator
    fkpl_arrays = fokker_planck_backward_euler_data(
                        mass, zeds,
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
        Fgeneral = allocate_float(vpa.n,vperp.n,1,1,species.n)
        @views Fold = Fgeneral[:,:,1,1,:]
    else
        Fold = allocate_float(vpa.n,vperp.n,species.n)
    end
    CC = fkpl_arrays.CCs # needed for dSdt diagnostic
    # dummy arrays needed for diagnostics
    Fout = allocate_float(vpa.n,vperp.n,species.n,2)
    Fdummy1 = allocate_float(vpa.n,vperp.n,species.n)
    Fdummy2 = allocate_float(vpa.n,vperp.n)
    Fdummy3 = allocate_float(vpa.n,vperp.n)
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
    # print diagnostic info to screen
    if print_diagnostics
        diagnose_F_Maxwellian(CC, Fold,Fdummy1,Fdummy2,Fdummy3,fkpl_arrays.fp_operator,moments,time,0)
    end
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
        if print_diagnostics
            diagnose_F_Maxwellian(CC,Fold,Fdummy1,Fdummy2,Fdummy3,fkpl_arrays.fp_operator,moments,time,it)
        end
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
        return pdf_and_grid(vpa.grid,vperp.grid,Fout)
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
    vth0=[0.5]::Vector{mk_float}, vperp0=[1.0]::Vector{mk_float}, vpa0=[0.0]::Vector{mk_float}, zbeam=[0.0]::Vector{mk_float},
    # source info
    source_rate=[1.0]::Vector{mk_float}, source_v0=[1.0]::Vector{mk_float}, source_vth=[0.05]::Vector{mk_float}, sink_rate=[5.0]::Vector{mk_float}, sink_vth=0.05::mk_float, constant_sink=false::Bool,
    # species info
    mass=[1.0]::Vector{mk_float}, zeds=[2.0]::Vector{mk_float}, density_in=[1.0e-8]::Vector{mk_float},
    # grid info
    ngrid=5::mk_int, nelement_vpa=32::mk_int, nelement_vperp=16::mk_int,
    Lvpa=3.0::mk_float, Lvperp=1.5::mk_float,
    # boundary condition info
    bc_vpa=natural_boundary_condition::finite_element_boundary_condition_type,
    bc_vperp=natural_boundary_condition::finite_element_boundary_condition_type,
    # time advance info
    ntime=1::mk_int,delta_t=0.01::mk_float,
    # nonlinear solver options
    atol = 1.0e-10::mk_float, rtol = 0.0::mk_float,
    nonlinear_max_iterations = 20::mk_int, test_particle_preconditioner=true::Bool,
    # model options
    electron_mass = 1.0/1836.0::mk_float, thermal_temperature = 0.01::mk_float,
    test_linearised_advance=false::Bool,
    use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=false::Bool,
    test_numerical_conserving_terms=true::Bool,
    test_numerical_conserving_terms_on_C=true::Bool,
    boundary_data_option=multipole_expansion::boundary_data_type,
    multi_species_operator_option=single_assembly_per_species::multi_species_operator_type,
    print_diagnostics=true::Bool, print_timing=true::Bool, print_final_pdf=false::Bool)

    # number of species
    nspecies = length(zeds)
    # check inputs are consistent
    @boundscheck (nspecies == length(mass) && nspecies == length(vth0)
                   && nspecies == length(vperp0) && nspecies == length(vpa0)
                   && nspecies == length(zbeam) && nspecies == length(density_in)) || throw(BoundsError(zeds))
    # check density_in > 0
    for is in 1:nspecies
        if density_in[is] < 1.0e-14
            error("Require values of density_in to be greater than 0.999e-14")
        end
    end
    start_init_time = now()
    # group integer inputs using `scalar_coordinate_inputs` from FokkerPlanck.coordinates
    input_vpa = scalar_coordinate_inputs(ngrid, nelement_vpa, Lvpa)
    input_vperp = scalar_coordinate_inputs(ngrid, nelement_vperp, Lvperp)
    # fixed background Maxwellian inputs
    # parameters of fixed background species
    msp = [0.25*electron_mass, 0.25]
    Zsp = [-1.0, 1.0]
    denssp = [1.0,1.0]
    uparsp = [0.0,0.0]
    temp = thermal_temperature
    vthsp = [sqrt(2.0*temp/msp[1]), sqrt(2.0*temp/msp[2])]
    fixed_background_plasma_in = fixed_background_plasma_input(msp,Zsp,denssp,uparsp,vthsp)
    # information for sources of the evolving species
    source_data_in = slowing_down_source_data_input(source_rate,source_vth,source_v0,sink_rate,sink_vth,constant_sink)
    # initialise all arrays needed to evaluate the nonlinear Fokker-Planck operator
    fkpl_arrays = fokker_planck_backward_euler_data(
                        mass, zeds,
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
    Fold = allocate_float(vpa.n,vperp.n,species.n)
    CC = fkpl_arrays.CCs # needed for dSdt diagnostic
    # dummy arrays needed for diagnostics
    Fout = allocate_float(vpa.n,vperp.n,species.n,2)
    Fdummy1 = allocate_float(vpa.n,vperp.n,species.n)
    Fdummy2 = allocate_float(vpa.n,vperp.n)
    Fdummy3 = allocate_float(vpa.n,vperp.n)
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
    # print diagnostic info to screen
    if print_diagnostics
        diagnose_F_Maxwellian(CC, Fold,Fdummy1,Fdummy2,Fdummy3,fkpl_arrays.fp_operator,moments,time,0)
    end
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
        if print_diagnostics
            diagnose_F_Maxwellian(CC,Fold,Fdummy1,Fdummy2,Fdummy3,fkpl_arrays.fp_operator,moments,time,it,updated_CC=!test_linearised_advance)
        end
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
    pdfandgrid = pdf_and_grid(vpa.grid,vperp.grid,Fout)
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
