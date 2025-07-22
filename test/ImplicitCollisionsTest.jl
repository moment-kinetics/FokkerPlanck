using Dates
using FokkerPlanck.array_allocation: allocate_float
using FokkerPlanck.type_definitions: mk_float, mk_int
using FokkerPlanck: init_fokker_planck_collisions,fokker_planck_self_collisions_backward_euler_step!,
                                    fokker_planck_self_collision_operator_weak_form!,
                                    fokkerplanck_weakform_arrays_struct
using FokkerPlanck.fokker_planck_calculus: multipole_expansion, boundary_data_type

# provides functions for test below to keep this script concise
include(joinpath(@__DIR__,"ImplicitCollisionsTestBase.jl"))

function test_implicit_collisions(; 
    # initial pdf info
    vth0=0.5::mk_float, vperp0=1.0::mk_float, vpa0=0.0::mk_float, zbeam=0.0::mk_float,
    # grid info
    ngrid=3::mk_int, nelement_vpa=8::mk_int, nelement_vperp=4::mk_int,
    Lvpa=6.0::mk_float, Lvperp=3.0::mk_float,
    # boundary condition info
    bc_vpa="none"::String, bc_vperp="none"::String,
    # time advance info
    ntime=1::mk_int,delta_t=1.0::mk_float,
    # nonlinear solver options
    atol = 1.0e-10::mk_float, rtol = 0.0::mk_float,
    nonlinear_max_iterations = 20::mk_int, test_particle_preconditioner=true::Bool,
    # model options
    test_linearised_advance=false::Bool,
    use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=false::Bool,
    test_numerical_conserving_terms=false::Bool,
    boundary_data_option=multipole_expansion::boundary_data_type,
    test_external_chebyshev_grid=false::Bool,
    print_diagnostics=true::Bool, print_timing=true::Bool,
    # if ci test, return initial and final pdfs for regression testing
    continuous_integration_test=false::Bool,
    # if external user, may pass a slice of an array into functions
    test_input_array_type=false::Bool)
    
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
    fkpl_arrays = init_fokker_planck_collisions(
                        input_vpa,
                        input_vperp;
                        bc_vpa=bc_vpa,
                        bc_vperp=bc_vperp,
                        boundary_data_option=boundary_data_option,
                        nl_solver_atol=atol,
                        nl_solver_rtol=rtol,
                        nl_solver_nonlinear_max_iterations=nonlinear_max_iterations,
                        print_to_screen=print_diagnostics)
    # extract coordinates
    vpa = fkpl_arrays.vpa
    vperp = fkpl_arrays.vperp
    # arrays needed for advance
    if test_input_array_type
        Fgeneral = allocate_float(vpa.n,vperp.n,1,1,1)
        @views Fold = Fgeneral[:,:,1,1,1]
    else
        Fold = allocate_float(vpa.n,vperp.n)
    end
    # dummy arrays needed for diagnostics
    Fout = allocate_float(vpa.n,vperp.n,2)
    Fdummy1 = allocate_float(vpa.n,vperp.n)
    Fdummy2 = allocate_float(vpa.n,vperp.n)
    Fdummy3 = allocate_float(vpa.n,vperp.n)
    # physics parameters
    ms = 1.0
    nuss = 1.0
    # initial condition
    time = 0.0
    set_initial_pdf!(Fold,vpa,vperp,vpa0,vperp0,vth0,zbeam)
    # store initial pdf for output
    Fout[:,:,1] .= Fold
    # get initial C[F,F] for entropy production diagnostic
    fokker_planck_self_collision_operator_weak_form!(Fold, ms, nuss, fkpl_arrays,
                use_conserving_corrections=test_numerical_conserving_terms)
    # print diagnostic info to screen
    if print_diagnostics
        diagnose_F_Maxwellian(Fold,Fdummy1,Fdummy2,Fdummy3,fkpl_arrays,time,ms,0)
    end
    finish_init_time = now()
    # time advance with backward Euler
    for it in 1:ntime
        # use Fold = F^n to obtain Fnew = F^n+1 for n = it 
        fokker_planck_self_collisions_backward_euler_step!(Fold, delta_t, ms, nuss, fkpl_arrays,
            test_numerical_conserving_terms=test_numerical_conserving_terms,
            test_particle_preconditioner=test_particle_preconditioner,
            test_linearised_advance=test_linearised_advance,
            use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=use_Maxwellian_Rosenbluth_coefficients_in_preconditioner)
        # update the pdf by extracting Fnew from fkpl_arrays
        Fnew = fkpl_arrays.Fnew
        @inbounds begin
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    Fold[ivpa,ivperp] = Fnew[ivpa,ivperp]
                end
            end
        end
        # diagnose the updated Fold
        time += delta_t
        if print_diagnostics
            diagnose_F_Maxwellian(Fold,Fdummy1,Fdummy2,Fdummy3,fkpl_arrays,time,ms,it)
        end
    end
    finish_run_time = now()
    # store final pdf for output
    Fout[:,:,2] .= Fold
    # println("total newton iterations: ", fkpl_arrays.nl_solver_data.nonlinear_iterations[])
    if print_timing
        # print some timing information
        println("init time (ms): ", Dates.value(finish_init_time - start_init_time))
        println("run time (ms): ", Dates.value(finish_run_time - finish_init_time))
    end
    # to make this function testable
    if continuous_integration_test
        # uncomment to update tests
        # print_grid(vpa)
        # print_grid(vperp)
        # print_pdf(Fout)
        return pdf_and_grid(vpa.grid,vperp.grid,Fout)
    else
        return nothing
    end
end
    
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")
    # run once to precompile
    test_implicit_collisions(test_particle_preconditioner=true,test_numerical_conserving_terms=true,
    vth0=0.5,vperp0=1.0,vpa0=1.0, nelement_vpa=4,nelement_vperp=2,Lvpa=8.0,Lvperp=4.0, bc_vpa="none", bc_vperp="none",
     ntime=1, delta_t = 1.0, ngrid=5, test_linearised_advance=false)
    # run a standard case now we are precompiled
    test_implicit_collisions(test_particle_preconditioner=true,test_numerical_conserving_terms=true,
    vth0=0.5,vperp0=1.0,vpa0=1.0, nelement_vpa=32,nelement_vperp=16,Lvpa=8.0,Lvperp=4.0, bc_vpa="none", bc_vperp="none",
     ntime=100, delta_t = 1.0, ngrid=5, test_linearised_advance=false)
end
