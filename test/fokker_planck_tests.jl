module FokkerPlanckTestsBase

using Test: @testset, @test

export backward_Euler_linearised_collisions_test
export backward_Euler_fokker_planck_self_collisions_test

using LinearAlgebra: mul!, ldiv!
using FokkerPlanck.array_allocation: allocate_float
using FokkerPlanck.coordinates: finite_element_coordinate, scalar_coordinate_inputs,
                                finite_element_boundary_condition_type,
                                natural_boundary_condition, zero_boundary_condition
using FokkerPlanck.type_definitions: mk_float, mk_int
using FokkerPlanck.velocity_moments: get_density, get_upar, get_pressure, get_ppar, get_pperp, get_qpar, get_rmom
using FokkerPlanck.fokker_planck_calculus: direct_integration, multipole_expansion, delta_f_multipole

using FokkerPlanck: init_fokker_planck_collisions, fokker_planck_collision_operator_weak_form!
using FokkerPlanck: conserving_corrections!
using FokkerPlanck: fokker_planck_cross_species_collision_operator_Maxwellian_Fsp!
using FokkerPlanck: fokker_planck_self_collisions_backward_euler_step!, calculate_entropy_production
using FokkerPlanck.fokker_planck_test: print_test_data, fkpl_error_data, allocate_error_data #, plot_test_data
using FokkerPlanck.fokker_planck_test: F_Maxwellian, G_Maxwellian, H_Maxwellian, F_Beam
using FokkerPlanck.fokker_planck_test: d2Gdvpa2_Maxwellian, d2Gdvperp2_Maxwellian, d2Gdvperpdvpa_Maxwellian, dGdvperp_Maxwellian
using FokkerPlanck.fokker_planck_test: dHdvperp_Maxwellian, dHdvpa_Maxwellian, Cssp_Maxwellian_inputs
using FokkerPlanck.fokker_planck_calculus: calculate_rosenbluth_potentials_via_elliptic_solve!, calculate_rosenbluth_potential_boundary_data_exact!
using FokkerPlanck.fokker_planck_calculus: test_rosenbluth_potential_boundary_data, rosenbluth_potential_boundary_data
using FokkerPlanck.fokker_planck_calculus: enforce_vpavperp_BCs!, calculate_rosenbluth_potentials_via_direct_integration!
using FokkerPlanck.fokker_planck_calculus: interpolate_2D_vspace!, calculate_test_particle_preconditioner!
using FokkerPlanck.fokker_planck_calculus: advance_linearised_test_particle_collisions!, fokkerplanck_weakform_arrays_struct,
                                            fokkerplanck_arrays_direct_integration_struct

function create_grids(ngrid,nelement_vpa,nelement_vperp;
                      Lvpa=12.0,Lvperp=6.0,bc_vpa=zero_boundary_condition,bc_vperp=zero_boundary_condition)

        # create the 'input' struct containing input info needed to create a
        # coordinate
        element_spacing_option = "uniform"
        # create the coordinate structs
        vperp = finite_element_coordinate("vperp",
                                scalar_coordinate_inputs(ngrid,
                                    nelement_vperp,
                                    Lvperp),
                                element_spacing_option=element_spacing_option,
                                bc=bc_vperp)
        vpa = finite_element_coordinate("vpa",
                                scalar_coordinate_inputs(ngrid,
                                    nelement_vpa,
                                    Lvpa),
                                    element_spacing_option=element_spacing_option,
                                    bc=bc_vpa)
        
        return vpa, vperp
end

# test of preconditioner matrix for nonlinear implicit solve.
# We use the preconditioner matrix for a time advance of
# dF/dt = C[F,F_M], with F_M a fixed Maxwellian distribution.
# We test that the result F is close to F_M.
function backward_Euler_linearised_collisions_test(;      
                # grid and physics parameters
                ngrid = 5,
                nelement_vpa = 16,
                nelement_vperp = 8,
                bc_vpa=natural_boundary_condition,
                bc_vperp=natural_boundary_condition,
                ms = 1.0,
                delta_t = 1.0,
                nuss = 1.0,
                ntime = 100,
                # background Maxwellian
                dens = 1.0,
                upar = 0.0,
                vth = 1.0,
                # initial beam parameters
                vpa0 = 1.0,
                vperp0 = 1.0,
                vth0 = 0.5,
                # options
                boundary_data_option = multipole_expansion,
                use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=true,
                print_to_screen=false,
                # error tolerances
                atol_max = 2.0e-5,
                atol_L2 = 2.0e-6,
                atol_dens = 1.0e-8,
                atol_upar = 1.0e-10,
                atol_vth = 1.0e-7)

    # initialise arrays
    vpa, vperp = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                                                Lvpa=10.0,Lvperp=5.0,
                                                                bc_vperp=bc_vperp,bc_vpa=bc_vpa)
    fkpl_arrays = fokkerplanck_weakform_arrays_struct(vpa,vperp,boundary_data_option,
                        print_to_screen=print_to_screen)
    dummy_array = allocate_float(vpa.n,vperp.n)
    FMaxwell = allocate_float(vpa.n,vperp.n)
    FMaxwell_err = allocate_float(vpa.n,vperp.n)
    # make sure to use anyv communicator for any array that is modified in fokker_planck.jl functions
    pdf = allocate_float(vpa.n,vperp.n)
    
    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                FMaxwell[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                pdf[ivpa,ivperp] = F_Beam(vpa0,vperp0,vth0,vpa,vperp,ivpa,ivperp)
            end
        end
    end
    
    # normalise to unit density
    @views densfac = get_density(pdf,vpa,vperp)
    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                pdf[ivpa,ivperp] /= densfac
            end
        end
    end
    # calculate the linearised advance matrix 
    calculate_test_particle_preconditioner!(FMaxwell,delta_t,ms,ms,nuss,fkpl_arrays,
        use_Maxwellian_Rosenbluth_coefficients=use_Maxwellian_Rosenbluth_coefficients_in_preconditioner)
    for it in 1:ntime
        advance_linearised_test_particle_collisions!(pdf,fkpl_arrays)
    end
    # now check distribution
    test_F_Maxwellian(FMaxwell,pdf,
            vpa,vperp,
            FMaxwell_err,dummy_array, 
            dens, upar, vth, ms,
            atol_max, atol_L2,
            atol_dens, atol_upar, atol_vth, 
            print_to_screen=print_to_screen)
    
    return nothing
end

function test_F_Maxwellian(pdf_Maxwell,pdf,
    vpa,vperp,
    dummy_array_1,dummy_array_2, 
    dens, upar, vth, mass,
    atol_max, atol_L2,
    atol_dens, atol_upar, atol_vth; 
    print_to_screen=false)
    
    F_M_max, F_M_L2 = print_test_data(pdf_Maxwell,pdf,dummy_array_1,"pdf",
        vpa,vperp,dummy_array_2,print_to_screen=print_to_screen)
    dens_num = get_density(pdf, vpa, vperp)
    upar_num = get_upar(pdf, vpa, vperp, dens)
    pressure = get_pressure(pdf, vpa, vperp, upar)
    vth_num = sqrt(2.0*pressure/(dens*mass))
    @test F_M_max < atol_max
    @test F_M_L2 < atol_L2
    @test abs(dens_num - dens) < atol_dens
    @test abs(upar_num - upar) < atol_upar
    @test abs(vth_num - vth) < atol_vth
    return nothing
end

function diagnose_F_Maxwellian(pdf,pdf_exact,pdf_dummy_1,pdf_dummy_2,vpa,vperp,time,mass,it)
    
    dens = get_density(pdf,vpa,vperp)
    upar = get_upar(pdf, vpa, vperp, dens)
    pressure = get_pressure(pdf, vpa, vperp, upar)
    vth = sqrt(2.0*pressure/(dens*mass))
    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                pdf_exact[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            end
        end
    end
    println("it = ", it, " time: ", time)
    print_test_data(pdf_exact,pdf,pdf_dummy_1,"F",vpa,vperp,pdf_dummy_2;print_to_screen=true)
    println("dens: ", dens)
    println("upar: ", upar)
    println("vth: ", vth)
    if vpa.bc == zero_boundary_condition
        println("test vpa bc: F[1, :]", pdf[1, :])
        println("test vpa bc: F[end, :]", pdf[end, :])
    end
    if vperp.bc == zero_boundary_condition
        println("test vperp bc: F[:, end]", pdf[:, end])
    end
    return nothing
end

# Test of implementation of backward Euler solve of d F / d t = C[F, F]
# i.e., we solve F^n+1 - F^n = delta_t * C[ F^n+1, F^n+1 ]
# using a Newton-Krylov root-finding method. This test function
# can be used to check the performance of the solver at a single
# velocity space point. We initialise with a beam distribution
# ~ exp ( - ((vpa - vpa0)^2 + (vperp - vperp0)^2) / vth0^2 )
# and timestep for a fixed timestep delta_t to a maximum time
# ntime * delta_t. Errors between F and F_Maxwellian can be printed to screen.
# Different algorithm options can be checked.
function backward_Euler_fokker_planck_self_collisions_test(; 
    # initial beam parameters 
    vth0=0.5,
    vperp0=1.0,
    vpa0=1.0,
    # grid parameters
    ngrid=5,
    nelement_vpa=16,
    nelement_vperp=8,
    Lvpa=10.0,
    Lvperp=5.0,
    bc_vpa=natural_boundary_condition,
    bc_vperp=natural_boundary_condition,
    # timestepping parameters
    ntime=100,
    delta_t=1.0,
    # options
    test_particle_preconditioner=true,
    test_linearised_advance=false,
    use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=false,
    test_numerical_conserving_terms=true,
    boundary_data_option=multipole_expansion,
    print_to_screen=true,
    # error tolerances
    atol_max = 2.0e-5,
    atol_L2 = 2.0e-6,
    atol_dens = 1.0e-8,
    atol_upar = 5.0e-9,
    atol_vth = 1.0e-7)
    
    vpa, vperp = create_grids(ngrid,nelement_vpa,nelement_vperp;
                      Lvpa=Lvpa,Lvperp=Lvperp,bc_vpa=bc_vpa,bc_vperp=bc_vperp)
    fkpl_arrays = fokkerplanck_weakform_arrays_struct(vpa,vperp,boundary_data_option;
                        nl_solver_atol=1.0e-10,
                        nl_solver_rtol=0.0,
                        print_to_screen=print_to_screen)
    
    # initial condition
    Fold = allocate_float(vpa.n,vperp.n)
    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                Fold[ivpa,ivperp] = F_Beam(vpa0,vperp0,vth0,vpa,vperp,ivpa,ivperp)
            end
        end
    end
    if vpa.bc == zero_boundary_condition
        @inbounds for ivperp in 1:vperp.n
            Fold[1,ivperp] = 0.0
            Fold[end,ivperp] = 0.0
        end
    end
    if vperp.bc == zero_boundary_condition
        @inbounds for ivpa in 1:vpa.n
            Fold[ivpa,end] = 0.0
        end
    end
    # normalise to unit density
    @views densfac = get_density(Fold[:,:],vpa,vperp)
    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                Fold[ivpa,ivperp] /= densfac
            end
        end
    end
    # dummy arrays
    Fdummy1 = allocate_float(vpa.n,vperp.n)
    Fdummy2 = allocate_float(vpa.n,vperp.n)
    Fdummy3 = allocate_float(vpa.n,vperp.n)
    FMaxwell = allocate_float(vpa.n,vperp.n)
    # physics parameters
    ms = 1.0
    nuss = 1.0
    
    # initial condition 
    time = 0.0
    # Maxwellian and parameters
    dens = get_density(Fold,vpa,vperp)
    upar = get_upar(Fold, vpa, vperp, dens)
    pressure = get_pressure(Fold, vpa, vperp, upar)
    vth = sqrt(2.0*pressure/(dens*ms))
    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                FMaxwell[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            end
        end
    end
    
    if print_to_screen
        diagnose_F_Maxwellian(Fold,Fdummy1,Fdummy2,Fdummy3,vpa,vperp,time,ms,0)
    end
    for it in 1:ntime
        fokker_planck_self_collisions_backward_euler_step!(Fold, delta_t, ms, nuss, fkpl_arrays,
            use_conserving_corrections=test_numerical_conserving_terms,
            test_particle_preconditioner=test_particle_preconditioner,
            test_linearised_advance=test_linearised_advance,
            use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=use_Maxwellian_Rosenbluth_coefficients_in_preconditioner)
        # update the pdf
        Fnew = fkpl_arrays.Fnew
        @inbounds begin
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    Fold[ivpa,ivperp] = Fnew[ivpa,ivperp]
                end
            end
        end
        # diagnose Fold
        time += delta_t
        if print_to_screen
            diagnose_F_Maxwellian(Fold,Fdummy1,Fdummy2,Fdummy3,vpa,vperp,time,ms,it)
        end
    end
    
    # now check distribution
    test_F_Maxwellian(FMaxwell,Fold,
            vpa,vperp,
            Fdummy2,Fdummy3, 
            dens, upar, vth, ms,
            atol_max, atol_L2,
            atol_dens, atol_upar, atol_vth, 
            print_to_screen=print_to_screen)
    
    return nothing
end

function numerical_error_corrections_test(; 
    ngrid = 5, # chosen for a quick test -- direct integration is slow!
    nelement_vpa = 8,
    nelement_vperp = 4,
    Lvpa = 12.0,
    Lvperp = 6.0,
    abeam = 0.5,
    vpa0 = 1.0,
    vperp0 = 1.0,
    vth0 = 0.5,
    atol = 1.0e-14,
    print_to_screen=false,
    )
    vpa, vperp = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                                                Lvpa=Lvpa,Lvperp=Lvperp)
    boundary_data_option = multipole_expansion
    fkpl_arrays = fokkerplanck_weakform_arrays_struct(vpa,vperp,boundary_data_option,
                                        print_to_screen=print_to_screen)

    pdf_in = allocate_float(vpa.n,vperp.n)
    C_num = allocate_float(vpa.n,vperp.n)
    denss, upars, vths = 1.0, 1.0, 1.0
    # initialise a distribution that has a qpar
    for ivperp in 1:vperp.n
        for ivpa in 1:vpa.n
            pdf_in[ivpa,ivperp] = (abeam * F_Beam(vpa0,vperp0,vth0,vpa,vperp,ivpa,ivperp)
                                   + F_Beam(0.0,vperp0,vth0,vpa,vperp,ivpa,ivperp))
        end
    end
    dens = get_density(pdf_in, vpa, vperp)
    upar = get_upar(pdf_in, vpa, vperp, dens)
    pressure = get_pressure(pdf_in, vpa, vperp, upar)
    vth = sqrt(2.0*pressure/dens)
    ppar = get_ppar(pdf_in, vpa, vperp, upar)
    qpar = get_qpar(pdf_in, vpa, vperp, upar)
    rmom = get_rmom(pdf_in, vpa, vperp, upar)
    # check test pdf unchanged
    if abeam == 0.5 && vpa0 == 1.0 && vperp0 == 1.0 && vth0 == 0.5
        @test isapprox(dens, 7.416900452984803, atol=atol)
        @test isapprox(upar, 0.33114644602432997, atol=atol)
        @test isapprox(vth, 1.0695323945144575, atol=atol)
        @test isapprox(qpar, 0.29147880412034594, atol=atol)
        @test isapprox(rmom, 27.57985752143237, atol=3*atol)
    end

    CC = fkpl_arrays.CC
    # fill CC with a pdf that definitely has a density, mean flow, and pressure (unlike C[F,F])
    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                CC[ivpa,ivperp] = F_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
            end
        end
    end
    # make ad-hoc conserving corrections to remove the denisty, mean flow, and pressure
    conserving_corrections!(CC,pdf_in,vpa,vperp)
    
    # extract result
    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                C_num[ivpa,ivperp] = CC[ivpa,ivperp]
            end
        end
    end
    
    # check CC now has zero density, flow, and pressure moments
    dn = get_density(C_num, vpa, vperp)
    du = get_upar(C_num, vpa, vperp, 1.0)
    dp = get_pressure(C_num, vpa, vperp, upar)
    @test abs(dn) < atol
    @test abs(du) < atol
    @test abs(dp) < atol
    return nothing
end

function test_interpolate_2D_vspace(; ngrid=9,
                                nelement_vpa=16,
                                nelement_vperp = 8,
                                rtol = 3.0e-8)
    ngrid = 9
    nelement_vpa = 16
    nelement_vperp = 8
    vpa, vperp = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                Lvpa=8.0,Lvperp=4.0)

    # electron pdf on electron grids
    Fe = allocate_float(vpa.n,vperp.n)
    # electron pdf on ion normalised grids
    Fe_interp_ion_units = allocate_float(vpa.n,vperp.n)
    # exact value for comparison
    Fe_exact_ion_units = allocate_float(vpa.n,vperp.n)
    # ion pdf on ion grids
    Fi = allocate_float(vpa.n,vperp.n)
    # ion pdf on electron normalised grids
    Fi_interp_electron_units = allocate_float(vpa.n,vperp.n)
    # exact value for comparison
    Fi_exact_electron_units = allocate_float(vpa.n,vperp.n)
    # test array
    F_err = allocate_float(vpa.n,vperp.n)

    dense = 1.0
    upare = 0.0 # upare in electron reference units
    vthe = 1.0 # vthe in electron reference units
    densi = 1.0
    upari = 0.0 # upari in ion reference units
    vthi = 1.0 # vthi in ion reference units
    # reference speeds for electrons and ions
    cref_electron = 60.0
    cref_ion = 1.0
    # scale factor for change of reference speed
    scalefac = cref_ion/cref_electron

    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
            Fe[ivpa,ivperp] = F_Maxwellian(dense,upare,vthe,vpa,vperp,ivpa,ivperp)
            Fe_exact_ion_units[ivpa,ivperp] = F_Maxwellian(dense,upare/scalefac,vthe/scalefac,vpa,vperp,ivpa,ivperp)/(scalefac^3)
            Fi[ivpa,ivperp] = F_Maxwellian(densi,upari,vthi,vpa,vperp,ivpa,ivperp)
            Fi_exact_electron_units[ivpa,ivperp] = (scalefac^3)*F_Maxwellian(densi,upari*scalefac,vthi*scalefac,vpa,vperp,ivpa,ivperp)
            end
        end
    end

    interpolate_2D_vspace!(Fe_interp_ion_units,Fe,vpa,vperp,scalefac)
    #println("Fe",Fe)
    #println("Fe interp",Fe_interp_ion_units)
    #println("Fe exact",Fe_exact_ion_units)
    interpolate_2D_vspace!(Fi_interp_electron_units,Fi,vpa,vperp,1.0/scalefac)
    #println("Fi",Fi)
    #println("Fi interp", Fi_interp_electron_units)
    #println("Fi exact",Fi_exact_electron_units)

    # check the result
    # for electron data on ion grids
    @. F_err = abs(Fe_interp_ion_units - Fe_exact_ion_units)
    max_F_err = maximum(F_err)
    max_F = maximum(Fe_exact_ion_units)
    #println(max_F)
    @test max_F_err < rtol * max_F
    # for ion data on electron grids
    @. F_err = abs(Fi_interp_electron_units - Fi_exact_electron_units)
    max_F_err = maximum(F_err)
    max_F = maximum(Fi_exact_electron_units)
    #println(max_F)
    @test max_F_err < rtol * max_F

    return nothing
end

function runtests()
    print_to_screen = false
    @testset "Fokker Planck tests" begin
        println("Fokker Planck tests")
        @testset "backward-Euler nonlinear Fokker-Planck collisions" begin
            println("    - test backward-Euler nonlinear Fokker-Planck collisions")
            @testset "$bc" for bc in (natural_boundary_condition, zero_boundary_condition)  
                println("        -  bc=$bc")
                # here test that a Maxwellian initial condition remains Maxwellian,
                # i.e., we check the numerical Maxwellian is close to the analytical one.
                # This is faster and more stable than doing a relaxation from vperp0 /= 0.
                backward_Euler_fokker_planck_self_collisions_test(bc_vperp=bc, bc_vpa=bc,
                   ntime = 10, delta_t = 0.1,
                   vth0 = 1.0, vpa0 = 1.0, vperp0 = 0.0, 
                   print_to_screen=print_to_screen)
            end
        end

        @testset "Lagrange-polynomial 2D interpolation" begin
            println("    - test Lagrange-polynomial 2D interpolation")
            test_interpolate_2D_vspace()
        end

        @testset "weak-form 2D differentiation" begin
        # tests the correct definition of mass and stiffness matrices in 2D
            println("    - test weak-form 2D differentiation")

            ngrid = 9
            nelement_vpa = 8
            nelement_vperp = 4
            vpa, vperp = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                                    Lvpa=2.0,Lvperp=1.0)
            nc_global = vpa.n*vperp.n
            boundary_data_option = multipole_expansion
            fkpl_arrays = fokkerplanck_weakform_arrays_struct(vpa,vperp,boundary_data_option,
                                                    print_to_screen=print_to_screen)
            KKpar2D_with_BC_terms_sparse = fkpl_arrays.KKpar2D_with_BC_terms_sparse
            KKperp2D_with_BC_terms_sparse = fkpl_arrays.KKperp2D_with_BC_terms_sparse
            lu_obj_MM = fkpl_arrays.lu_obj_MM

            dummy_array = allocate_float(vpa.n,vperp.n)
            fvpavperp = allocate_float(vpa.n,vperp.n)
            fvpavperp_test = allocate_float(vpa.n,vperp.n)
            fvpavperp_err = allocate_float(vpa.n,vperp.n)
            d2fvpavperp_dvpa2_exact = allocate_float(vpa.n,vperp.n)
            d2fvpavperp_dvpa2_err = allocate_float(vpa.n,vperp.n)
            d2fvpavperp_dvpa2_num = allocate_float(vpa.n,vperp.n)
            d2fvpavperp_dvperp2_exact = allocate_float(vpa.n,vperp.n)
            d2fvpavperp_dvperp2_err = allocate_float(vpa.n,vperp.n)
            d2fvpavperp_dvperp2_num = allocate_float(vpa.n,vperp.n)
            dfc = allocate_float(nc_global)
            dgc = allocate_float(nc_global)
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    fvpavperp[ivpa,ivperp] = exp(-vpa.grid[ivpa]^2 - vperp.grid[ivperp]^2)
                    d2fvpavperp_dvpa2_exact[ivpa,ivperp] = (4.0*vpa.grid[ivpa]^2 - 2.0)*exp(-vpa.grid[ivpa]^2 - vperp.grid[ivperp]^2)
                    # acutally d(vperp d f dvperp)/dvperp
                    d2fvpavperp_dvperp2_exact[ivpa,ivperp] = 4.0*(vperp.grid[ivperp]^2 - 1.0)*exp(-vpa.grid[ivpa]^2 - vperp.grid[ivperp]^2)
                end
            end

            # Make 1d views
            fc = vec(fvpavperp)
            d2fc_dvpa2 = vec(d2fvpavperp_dvpa2_num)
            d2fc_dvperp2 = vec(d2fvpavperp_dvperp2_num)

            #print_vector(fc,"fc",nc_global)
            # multiply by KKpar2D and fill dfc
            mul!(dfc,KKpar2D_with_BC_terms_sparse,fc)
            mul!(dgc,KKperp2D_with_BC_terms_sparse,fc)
            # invert mass matrix
            ldiv!(d2fc_dvpa2, lu_obj_MM, dfc)
            ldiv!(d2fc_dvperp2, lu_obj_MM, dgc)
            #print_vector(fc,"fc",nc_global)
            d2fvpavperp_dvpa2_max, d2fvpavperp_dvpa2_L2 = print_test_data(d2fvpavperp_dvpa2_exact,d2fvpavperp_dvpa2_num,d2fvpavperp_dvpa2_err,"d2fdvpa2",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
            @test d2fvpavperp_dvpa2_max < 1.0e-7
            @test d2fvpavperp_dvpa2_L2 < 1.0e-8
            d2fvpavperp_dvperp2_max, d2fvpavperp_dvperp2_L2 = print_test_data(d2fvpavperp_dvperp2_exact,d2fvpavperp_dvperp2_num,d2fvpavperp_dvperp2_err,"d2fdvperp2",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
            @test d2fvpavperp_dvperp2_max < 1.0e-7
            @test d2fvpavperp_dvperp2_L2 < 1.0e-8
            #if plot_test_output
            #    plot_test_data(d2fvpavperp_dvpa2_exact,d2fvpavperp_dvpa2_num,d2fvpavperp_dvpa2_err,"d2fvpavperp_dvpa2",vpa,vperp)
            #    plot_test_data(d2fvpavperp_dvperp2_exact,d2fvpavperp_dvperp2_num,d2fvpavperp_dvperp2_err,"d2fvpavperp_dvperp2",vpa,vperp)
            #end        
        end

        @testset "weak-form Rosenbluth potential calculation: elliptic solve" begin
            println("    - test weak-form Rosenbluth potential calculation: elliptic solve")
            @testset "$boundary_data_option" for boundary_data_option in (direct_integration,multipole_expansion,delta_f_multipole)
                println("        -  boundary_data_option=$boundary_data_option")
                ngrid = 9
                nelement_vpa = 8
                nelement_vperp = 4
                vpa, vperp = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                            Lvpa=12.0,Lvperp=6.0)
                
                fkpl_arrays = fokkerplanck_weakform_arrays_struct(vpa,vperp,boundary_data_option,
                                                                      print_to_screen=print_to_screen)
                dummy_array = allocate_float(vpa.n,vperp.n)
                F_M = allocate_float(vpa.n,vperp.n)
                H_M_exact = allocate_float(vpa.n,vperp.n)
                H_M_num = allocate_float(vpa.n,vperp.n)
                H_M_err = allocate_float(vpa.n,vperp.n)
                G_M_exact = allocate_float(vpa.n,vperp.n)
                G_M_num = allocate_float(vpa.n,vperp.n)
                G_M_err = allocate_float(vpa.n,vperp.n)
                d2Gdvpa2_M_exact = allocate_float(vpa.n,vperp.n)
                d2Gdvpa2_M_num = allocate_float(vpa.n,vperp.n)
                d2Gdvpa2_M_err = allocate_float(vpa.n,vperp.n)
                d2Gdvperp2_M_exact = allocate_float(vpa.n,vperp.n)
                d2Gdvperp2_M_num = allocate_float(vpa.n,vperp.n)
                d2Gdvperp2_M_err = allocate_float(vpa.n,vperp.n)
                dGdvperp_M_exact = allocate_float(vpa.n,vperp.n)
                dGdvperp_M_num = allocate_float(vpa.n,vperp.n)
                dGdvperp_M_err = allocate_float(vpa.n,vperp.n)
                d2Gdvperpdvpa_M_exact = allocate_float(vpa.n,vperp.n)
                d2Gdvperpdvpa_M_num = allocate_float(vpa.n,vperp.n)
                d2Gdvperpdvpa_M_err = allocate_float(vpa.n,vperp.n)
                dHdvpa_M_exact = allocate_float(vpa.n,vperp.n)
                dHdvpa_M_num = allocate_float(vpa.n,vperp.n)
                dHdvpa_M_err = allocate_float(vpa.n,vperp.n)
                dHdvperp_M_exact = allocate_float(vpa.n,vperp.n)
                dHdvperp_M_num = allocate_float(vpa.n,vperp.n)
                dHdvperp_M_err = allocate_float(vpa.n,vperp.n)

                dens, upar, vth = 1.0, 1.0, 1.0
                
                for ivperp in 1:vperp.n
                    for ivpa in 1:vpa.n
                        F_M[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                        H_M_exact[ivpa,ivperp] = H_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                        G_M_exact[ivpa,ivperp] = G_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                        d2Gdvpa2_M_exact[ivpa,ivperp] = d2Gdvpa2_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                        d2Gdvperp2_M_exact[ivpa,ivperp] = d2Gdvperp2_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                        dGdvperp_M_exact[ivpa,ivperp] = dGdvperp_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                        d2Gdvperpdvpa_M_exact[ivpa,ivperp] = d2Gdvperpdvpa_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                        dHdvpa_M_exact[ivpa,ivperp] = dHdvpa_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                        dHdvperp_M_exact[ivpa,ivperp] = dHdvperp_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                    end
                end
                rpbd_exact = rosenbluth_potential_boundary_data(vpa,vperp)
                # use known test function to provide exact data
                
                calculate_rosenbluth_potential_boundary_data_exact!(rpbd_exact,
                      H_M_exact,dHdvpa_M_exact,dHdvperp_M_exact,G_M_exact,
                      dGdvperp_M_exact,d2Gdvperp2_M_exact,
                      d2Gdvperpdvpa_M_exact,d2Gdvpa2_M_exact,vpa,vperp)
                # calculate the potentials numerically
                calculate_rosenbluth_potentials_via_elliptic_solve!(
                     fkpl_arrays.GG, fkpl_arrays.HH, fkpl_arrays.dHdvpa, fkpl_arrays.dHdvperp,
                     fkpl_arrays.d2Gdvpa2, fkpl_arrays.dGdvperp, fkpl_arrays.d2Gdvperpdvpa,
                     fkpl_arrays.d2Gdvperp2, F_M, vpa, vperp,
                     fkpl_arrays; algebraic_solve_for_d2Gdvperp2=false,
                     calculate_GG=true, calculate_dGdvperp=true)
                # extract C[Fs,Fs'] result
                # and Rosenbluth potentials for testing
                
                
                @inbounds begin
                    for ivperp in 1:vperp.n
                        for ivpa in 1:vpa.n
                        G_M_num[ivpa,ivperp] = fkpl_arrays.GG[ivpa,ivperp]
                        H_M_num[ivpa,ivperp] = fkpl_arrays.HH[ivpa,ivperp]
                        dHdvpa_M_num[ivpa,ivperp] = fkpl_arrays.dHdvpa[ivpa,ivperp]
                        dHdvperp_M_num[ivpa,ivperp] = fkpl_arrays.dHdvperp[ivpa,ivperp]
                        dGdvperp_M_num[ivpa,ivperp] = fkpl_arrays.dGdvperp[ivpa,ivperp]
                        d2Gdvperp2_M_num[ivpa,ivperp] = fkpl_arrays.d2Gdvperp2[ivpa,ivperp]
                        d2Gdvpa2_M_num[ivpa,ivperp] = fkpl_arrays.d2Gdvpa2[ivpa,ivperp]
                        d2Gdvperpdvpa_M_num[ivpa,ivperp] = fkpl_arrays.d2Gdvperpdvpa[ivpa,ivperp]
                        end
                    end
                end
                
                # test the boundary data
                max_H_boundary_data_err, max_dHdvpa_boundary_data_err,
                max_dHdvperp_boundary_data_err, max_G_boundary_data_err,
                max_dGdvperp_boundary_data_err, max_d2Gdvperp2_boundary_data_err,
                max_d2Gdvperpdvpa_boundary_data_err, max_d2Gdvpa2_boundary_data_err = test_rosenbluth_potential_boundary_data(fkpl_arrays.rpbd,rpbd_exact,vpa,vperp,print_to_screen=print_to_screen)
                if boundary_data_option==multipole_expansion
                    atol_max_H = 5.0e-8
                    atol_max_dHdvpa = 5.0e-8
                    atol_max_dHdvperp = 5.0e-8
                    atol_max_G = 5.0e-7
                    atol_max_dGdvperp = 5.0e-7
                    atol_max_d2Gdvperp2 = 5.0e-7
                    atol_max_d2Gdvperpdvpa = 5.0e-7
                    atol_max_d2Gdvpap2 = 1.0e-6
                else
                    atol_max_H = 2.0e-12
                    atol_max_dHdvpa = 2.0e-11
                    atol_max_dHdvperp = 6.0e-9
                    atol_max_G = 1.0e-11
                    atol_max_dGdvperp = 2.0e-7
                    atol_max_d2Gdvperp2 = 5.0e-8
                    atol_max_d2Gdvperpdvpa = 2.0e-8
                    atol_max_d2Gdvpap2 = 1.0e-11
                end
                @test max_H_boundary_data_err < atol_max_H
                @test max_dHdvpa_boundary_data_err < atol_max_dHdvpa
                @test max_dHdvperp_boundary_data_err < atol_max_dHdvperp
                @test max_G_boundary_data_err < atol_max_G
                @test max_dGdvperp_boundary_data_err < atol_max_dGdvperp
                @test max_d2Gdvperp2_boundary_data_err < atol_max_d2Gdvperp2
                @test max_d2Gdvperpdvpa_boundary_data_err < atol_max_d2Gdvperpdvpa
                @test max_d2Gdvpa2_boundary_data_err < atol_max_d2Gdvpap2
                # test the elliptic solvers
                H_M_max, H_M_L2 = print_test_data(H_M_exact,H_M_num,H_M_err,"H_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                dHdvpa_M_max, dHdvpa_M_L2 = print_test_data(dHdvpa_M_exact,dHdvpa_M_num,dHdvpa_M_err,"dHdvpa_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                dHdvperp_M_max, dHdvperp_M_L2 = print_test_data(dHdvperp_M_exact,dHdvperp_M_num,dHdvperp_M_err,"dHdvperp_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                G_M_max, G_M_L2 = print_test_data(G_M_exact,G_M_num,G_M_err,"G_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                d2Gdvpa2_M_max, d2Gdvpa2_M_L2 = print_test_data(d2Gdvpa2_M_exact,d2Gdvpa2_M_num,d2Gdvpa2_M_err,"d2Gdvpa2_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                dGdvperp_M_max, dGdvperp_M_L2 = print_test_data(dGdvperp_M_exact,dGdvperp_M_num,dGdvperp_M_err,"dGdvperp_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                d2Gdvperpdvpa_M_max, d2Gdvperpdvpa_M_L2 = print_test_data(d2Gdvperpdvpa_M_exact,d2Gdvperpdvpa_M_num,d2Gdvperpdvpa_M_err,"d2Gdvperpdvpa_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                d2Gdvperp2_M_max, d2Gdvperp2_M_L2 = print_test_data(d2Gdvperp2_M_exact,d2Gdvperp2_M_num,d2Gdvperp2_M_err,"d2Gdvperp2_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                if boundary_data_option==multipole_expansion
                    atol_max_H = 2.0e-7
                    atol_L2_H = 5.0e-9
                    atol_max_dHdvpa = 2.0e-6
                    atol_L2_dHdvpa = 5.0e-8
                    atol_max_dHdvperp = 2.0e-5
                    atol_L2_dHdvperp = 1.0e-7
                    atol_max_G = 5.0e-7
                    atol_L2_G = 5.0e-8
                    atol_max_d2Gdvpap2 = 1.0e-6
                    atol_L2_d2Gdvpa2 = 5.0e-8
                    atol_max_dGdvperp = 2.0e-6
                    atol_L2_dGdvperp = 2.0e-7
                    atol_max_d2Gdvperpdvpa = 2.0e-6
                    atol_L2_d2Gdvperpdvpa = 5.0e-8
                    atol_max_d2Gdvperp2 = 5.0e-7
                    atol_L2_d2Gdvperp2 = 5.0e-8
                else
                    atol_max_H = 2.0e-7
                    atol_L2_H = 5.0e-9
                    atol_max_dHdvpa = 2.0e-6
                    atol_L2_dHdvpa = 5.0e-8
                    atol_max_dHdvperp = 2.0e-5
                    atol_L2_dHdvperp = 1.0e-7
                    atol_max_G = 2.0e-8
                    atol_L2_G = 7.0e-10
                    atol_max_d2Gdvpap2 = 2.0e-7
                    atol_L2_d2Gdvpa2 = 4.0e-9
                    atol_max_dGdvperp = 2.0e-6
                    atol_L2_dGdvperp = 2.0e-7
                    atol_max_d2Gdvperpdvpa = 2.0e-6
                    atol_L2_d2Gdvperpdvpa = 2.0e-8
                    atol_max_d2Gdvperp2 = 3.0e-7
                    atol_L2_d2Gdvperp2 = 2.0e-8
                end
                @test H_M_max < atol_max_H
                @test H_M_L2 < atol_L2_H
                @test dHdvpa_M_max < atol_max_dHdvpa
                @test dHdvpa_M_L2 < atol_L2_dHdvpa
                @test dHdvperp_M_max < atol_max_dHdvperp
                @test dHdvperp_M_L2 < atol_L2_dHdvperp
                @test G_M_max < atol_max_G
                @test G_M_L2 < atol_L2_G
                @test d2Gdvpa2_M_max < atol_max_d2Gdvpap2
                @test d2Gdvpa2_M_L2 < atol_L2_d2Gdvpa2
                @test dGdvperp_M_max < atol_max_dGdvperp
                @test dGdvperp_M_L2 < atol_L2_dGdvperp
                @test d2Gdvperpdvpa_M_max < atol_max_d2Gdvperpdvpa
                @test d2Gdvperpdvpa_M_L2 < atol_L2_d2Gdvperpdvpa
                @test d2Gdvperp2_M_max < atol_max_d2Gdvperp2
                @test d2Gdvperp2_M_L2 < atol_L2_d2Gdvperp2
            end
        end

        @testset "weak-form collision operator calculation" begin
            println("    - test weak-form collision operator calculation")
            ngrid = 9
            nelement_vpa = 8
            nelement_vperp = 4
            vpa, vperp = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                    Lvpa=12.0,Lvperp=6.0)
            boundary_data_option=direct_integration
            fkpl_arrays = fokkerplanck_weakform_arrays_struct(vpa,vperp,boundary_data_option,
                                                        print_to_screen=print_to_screen)

            @testset "test_self_operator=$test_self_operator test_numerical_conserving_terms=$test_numerical_conserving_terms use_Maxwellian_Rosenbluth_coefficients=$use_Maxwellian_Rosenbluth_coefficients algebraic_solve_for_d2Gdvperp2=$algebraic_solve_for_d2Gdvperp2" for
                    (test_self_operator, test_numerical_conserving_terms,
                     use_Maxwellian_Rosenbluth_coefficients,
                     algebraic_solve_for_d2Gdvperp2) in ((true,false,false,false),(false,false,false,false),
                                                         (true,true,false,false),
                                                         (true,false,true,false),(true,false,false,true))

                dummy_array = allocate_float(vpa.n,vperp.n)
                Fs_M = allocate_float(vpa.n,vperp.n)
                F_M = allocate_float(vpa.n,vperp.n)
                C_M_num = allocate_float(vpa.n,vperp.n)
                C_M_exact = allocate_float(vpa.n,vperp.n)
                C_M_err = allocate_float(vpa.n,vperp.n)
                if test_self_operator
                    dens, upar, vth = 1.0, 1.0, 1.0
                    denss, upars, vths = dens, upar, vth
                else
                    denss, upars, vths = 1.0, -1.0, 2.0/3.0
                    dens, upar, vth = 1.0, 1.0, 1.0
                end
                ms = 1.0
                msp = 1.0
                nussp = 1.0
                for ivperp in 1:vperp.n
                    for ivpa in 1:vpa.n
                        Fs_M[ivpa,ivperp] = F_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
                        F_M[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                        C_M_exact[ivpa,ivperp] = Cssp_Maxwellian_inputs(denss,upars,vths,ms,
                                                                        dens,upar,vth,msp,
                                                                        nussp,vpa,vperp,ivpa,ivperp)
                    end
                end
                fokker_planck_collision_operator_weak_form!(Fs_M,F_M,ms,msp,nussp,fkpl_arrays,
                                                 use_Maxwellian_Rosenbluth_coefficients=use_Maxwellian_Rosenbluth_coefficients,
                                                 algebraic_solve_for_d2Gdvperp2=algebraic_solve_for_d2Gdvperp2,
                                                 calculate_GG = false, calculate_dGdvperp=false)
                if test_numerical_conserving_terms && test_self_operator
                    # enforce the boundary conditions on CC before it is used for timestepping
                    enforce_vpavperp_BCs!(fkpl_arrays.CC,vpa,vperp)
                    # make ad-hoc conserving corrections
                    conserving_corrections!(fkpl_arrays.CC,Fs_M,vpa,vperp)
                end
                # extract C[Fs,Fs'] result
                @inbounds begin
                    for ivperp in 1:vperp.n
                        for ivpa in 1:vpa.n
                            C_M_num[ivpa,ivperp] = fkpl_arrays.CC[ivpa,ivperp]
                        end
                    end
                end
                C_M_max, C_M_L2 = print_test_data(C_M_exact,C_M_num,C_M_err,"C_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                if test_self_operator && !test_numerical_conserving_terms && !use_Maxwellian_Rosenbluth_coefficients
                    atol_max = 6.0e-4/π^1.5
                    atol_L2 = 7.0e-6/π^1.5
                elseif test_self_operator && test_numerical_conserving_terms && !use_Maxwellian_Rosenbluth_coefficients
                    atol_max = 7.0e-4/π^1.5
                    atol_L2 = 7.0e-6/π^1.5
                elseif test_self_operator && !test_numerical_conserving_terms && use_Maxwellian_Rosenbluth_coefficients
                    atol_max = 8.0e-4/π^1.5
                    atol_L2 = 8.1e-6/π^1.5
                else
                    atol_max = 7.0e-2/π^1.5
                    atol_L2 = 6.0e-4/π^1.5
                end
                @test C_M_max < atol_max
                @test C_M_L2 < atol_L2

                # calculate the entropy production
                dSdt = calculate_entropy_production(Fs_M,fkpl_arrays)

                if test_self_operator && !test_numerical_conserving_terms
                    if algebraic_solve_for_d2Gdvperp2
                        rtol, atol = 0.0, 1.0e-7
                    else
                        rtol, atol = 0.0, 1.0e-8
                    end
                    @test isapprox(dSdt, rtol ; atol=atol)
                    delta_n = get_density(C_M_num, vpa, vperp)
                    delta_upar = get_upar(C_M_num, vpa, vperp, dens)
                    delta_pressure = msp*get_pressure(C_M_num, vpa, vperp, upar)
                    delta_ppar = msp*get_ppar(C_M_num, vpa, vperp, upar)
                    delta_pperp = get_pperp(delta_pressure, delta_ppar)
                    rtol, atol = 0.0, 1.0e-12
                    @test isapprox(delta_n, rtol ; atol=atol)
                    rtol, atol = 0.0, 1.0e-9
                    @test isapprox(delta_upar, rtol ; atol=atol)
                    if algebraic_solve_for_d2Gdvperp2
                        rtol, atol = 0.0, 1.0e-7*2
                    else
                        rtol, atol = 0.0, 1.0e-8*2
                    end
                    @test isapprox(delta_pressure, rtol ; atol=atol)
                    if print_to_screen
                        println("dSdt: $dSdt should be >0.0")
                        println("delta_n: ", delta_n)
                        println("delta_upar: ", delta_upar)
                        println("delta_pressure: ", delta_pressure)
                    end
                elseif test_self_operator && test_numerical_conserving_terms
                    rtol, atol = 0.0, 6.0e-7
                    @test isapprox(dSdt, rtol ; atol=atol)
                    delta_n = get_density(C_M_num, vpa, vperp)
                    delta_upar = get_upar(C_M_num, vpa, vperp, dens)
                    delta_pressure = msp*get_pressure(C_M_num, vpa, vperp, upar)
                    delta_ppar = msp*get_ppar(C_M_num, vpa, vperp, upar)
                    delta_pperp = get_pperp(delta_pressure, delta_ppar)
                    rtol, atol = 0.0, 1.0e-15
                    @test isapprox(delta_n, rtol ; atol=atol)
                    rtol, atol = 0.0, 1.0e-15
                    @test isapprox(delta_upar, rtol ; atol=atol)
                    rtol, atol = 0.0, 1.0e-15*2
                    @test isapprox(delta_pressure, rtol ; atol=atol)
                    if print_to_screen
                        println("dSdt: $dSdt should be >0.0")
                        println("delta_n: ", delta_n)
                        println("delta_upar: ", delta_upar)
                        println("delta_pressure: ", delta_pressure)
                    end
                else
                    atol = 1.0e-4
                    @test isapprox(dSdt, 4.090199753275297 ; atol=atol)
                    delta_n = get_density(C_M_num, vpa, vperp)
                    rtol, atol = 0.0, 1.0e-12
                    @test isapprox(delta_n, rtol ; atol=atol)
                    if print_to_screen
                        println("dSdt: $dSdt")
                        println("delta_n: ", delta_n)
                    end
                end
            end
            
        end

        @testset "weak-form (slowing-down) collision operator calculation" begin
            println("    - test weak-form (slowing-down) collision operator calculation")
            ngrid = 5
            nelement_vpa = 16
            nelement_vperp = 8
            vpa, vperp = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                                Lvpa=12.0,Lvperp=6.0)
            boundary_data_option=multipole_expansion
            fkpl_arrays = fokkerplanck_weakform_arrays_struct(vpa,vperp,boundary_data_option,
                                    print_to_screen=print_to_screen)

            @testset "slowing_down_test=true test_numerical_conserving_terms=$test_numerical_conserving_terms" for test_numerical_conserving_terms in (true,false)

                dummy_array = allocate_float(vpa.n,vperp.n)
                Fs_M = allocate_float(vpa.n,vperp.n)
                F_M = allocate_float(vpa.n,vperp.n)
                C_M_num = allocate_float(vpa.n,vperp.n)
                C_M_exact = allocate_float(vpa.n,vperp.n)
                C_M_err = allocate_float(vpa.n,vperp.n)

                # pick a set of parameters that represent slowing down
                # on slow ions and faster electrons, but which are close
                # enough to 1 for errors comparable to the self-collision operator
                # increasing or reducing vth, mass increases the errors
                dens, upar, vth = 1.0, 1.0, 1.0
                mref = 1.0
                Zref = 1.0
                msp = [1.0,0.2]#[0.25, 0.25/1836.0]
                Zsp = [0.5,0.5]#[0.5, 0.5]
                denssp = [1.0,1.0]#[1.0, 1.0]
                uparsp = [0.0,0.0]#[0.0, 0.0]
                vthsp = [sqrt(0.5/msp[1]), sqrt(0.5/msp[2])]#[sqrt(0.01/msp[1]), sqrt(0.01/msp[2])]
                nsprime = size(msp,1)
                nuref = 1.0

                for ivperp in 1:vperp.n
                    for ivpa in 1:vpa.n
                        Fs_M[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                        C_M_exact[ivpa,ivperp] = 0.0
                    end
                end
                # sum up contributions to cross-collision operator
                for isp in 1:nsprime
                    zfac = (Zsp[isp]/Zref)^2
                    nussp = nuref*zfac
                    for ivperp in 1:vperp.n
                        for ivpa in 1:vpa.n
                            C_M_exact[ivpa,ivperp] += Cssp_Maxwellian_inputs(dens,upar,vth,mref,
                                                                            denssp[isp],uparsp[isp],vthsp[isp],msp[isp],
                                                                            nussp,vpa,vperp,ivpa,ivperp)
                        end
                    end
                end
                fokker_planck_cross_species_collision_operator_Maxwellian_Fsp!(Fs_M,
                                     nuref,mref,Zref,msp,Zsp,denssp,uparsp,vthsp,
                                     fkpl_arrays;
                                     use_conserving_corrections=test_numerical_conserving_terms)
                # extract C[Fs,Fs'] result
                @inbounds begin
                    for ivperp in 1:vperp.n
                        for ivpa in 1:vpa.n
                            C_M_num[ivpa,ivperp] = fkpl_arrays.CC[ivpa,ivperp]
                        end
                    end
                end
                C_M_max, C_M_L2 = print_test_data(C_M_exact,C_M_num,C_M_err,"C_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                atol_max = 1.0e-3
                atol_L2 = 2.0e-5
                @test C_M_max < atol_max
                @test C_M_L2 < atol_L2
                if !test_numerical_conserving_terms
                    delta_n = get_density(C_M_num, vpa, vperp)
                    rtol, atol = 0.0, 1.0e-12
                    @test isapprox(delta_n, rtol ; atol=atol)
                    if print_to_screen
                        println("delta_n: ", delta_n)
                    end
                elseif test_numerical_conserving_terms
                    delta_n = get_density(C_M_num, vpa, vperp)
                    rtol, atol = 0.0, 1.0e-15
                    @test isapprox(delta_n, rtol ; atol=atol)
                    if print_to_screen
                        println("delta_n: ", delta_n)
                    end
                end
            end
            
        end

        @testset "weak-form Rosenbluth potential calculation: direct integration" begin
            println("    - test weak-form Rosenbluth potential calculation: direct integration")
            ngrid = 5 # chosen for a quick test -- direct integration is slow!
            nelement_vpa = 8
            nelement_vperp = 4
            vpa, vperp = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                            Lvpa=12.0,Lvperp=6.0)
            
            fkpl_arrays = fokkerplanck_arrays_direct_integration_struct(vperp,vpa;
                                                    print_to_screen=print_to_screen)
            dummy_array = allocate_float(vpa.n,vperp.n)
            F_M = allocate_float(vpa.n,vperp.n)
            H_M_exact = allocate_float(vpa.n,vperp.n)
            H_M_num = allocate_float(vpa.n,vperp.n)
            H_M_err = allocate_float(vpa.n,vperp.n)
            G_M_exact = allocate_float(vpa.n,vperp.n)
            G_M_num = allocate_float(vpa.n,vperp.n)
            G_M_err = allocate_float(vpa.n,vperp.n)
            d2Gdvpa2_M_exact = allocate_float(vpa.n,vperp.n)
            d2Gdvpa2_M_num = allocate_float(vpa.n,vperp.n)
            d2Gdvpa2_M_err = allocate_float(vpa.n,vperp.n)
            d2Gdvperp2_M_exact = allocate_float(vpa.n,vperp.n)
            d2Gdvperp2_M_num = allocate_float(vpa.n,vperp.n)
            d2Gdvperp2_M_err = allocate_float(vpa.n,vperp.n)
            dGdvperp_M_exact = allocate_float(vpa.n,vperp.n)
            dGdvperp_M_num = allocate_float(vpa.n,vperp.n)
            dGdvperp_M_err = allocate_float(vpa.n,vperp.n)
            d2Gdvperpdvpa_M_exact = allocate_float(vpa.n,vperp.n)
            d2Gdvperpdvpa_M_num = allocate_float(vpa.n,vperp.n)
            d2Gdvperpdvpa_M_err = allocate_float(vpa.n,vperp.n)
            dHdvpa_M_exact = allocate_float(vpa.n,vperp.n)
            dHdvpa_M_num = allocate_float(vpa.n,vperp.n)
            dHdvpa_M_err = allocate_float(vpa.n,vperp.n)
            dHdvperp_M_exact = allocate_float(vpa.n,vperp.n)
            dHdvperp_M_num = allocate_float(vpa.n,vperp.n)
            dHdvperp_M_err = allocate_float(vpa.n,vperp.n)

            dens, upar, vth = 1.0, 1.0, 1.0
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    F_M[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                    H_M_exact[ivpa,ivperp] = H_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                    G_M_exact[ivpa,ivperp] = G_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                    d2Gdvpa2_M_exact[ivpa,ivperp] = d2Gdvpa2_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                    d2Gdvperp2_M_exact[ivpa,ivperp] = d2Gdvperp2_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                    dGdvperp_M_exact[ivpa,ivperp] = dGdvperp_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                    d2Gdvperpdvpa_M_exact[ivpa,ivperp] = d2Gdvperpdvpa_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                    dHdvpa_M_exact[ivpa,ivperp] = dHdvpa_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                    dHdvperp_M_exact[ivpa,ivperp] = dHdvperp_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                end
            end
            # calculate the potentials numerically
            calculate_rosenbluth_potentials_via_direct_integration!(G_M_num,H_M_num,dHdvpa_M_num,dHdvperp_M_num,
             d2Gdvpa2_M_num,dGdvperp_M_num,d2Gdvperpdvpa_M_num,d2Gdvperp2_M_num,F_M,
             vpa,vperp,fkpl_arrays)
            # test the integration
            # to recalculate absolute tolerances atol, set print_to_screen = true
            H_M_max, H_M_L2 = print_test_data(H_M_exact,H_M_num,H_M_err,"H_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
            dHdvpa_M_max, dHdvpa_M_L2 = print_test_data(dHdvpa_M_exact,dHdvpa_M_num,dHdvpa_M_err,"dHdvpa_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
            dHdvperp_M_max, dHdvperp_M_L2 = print_test_data(dHdvperp_M_exact,dHdvperp_M_num,dHdvperp_M_err,"dHdvperp_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
            G_M_max, G_M_L2 = print_test_data(G_M_exact,G_M_num,G_M_err,"G_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
            d2Gdvpa2_M_max, d2Gdvpa2_M_L2 = print_test_data(d2Gdvpa2_M_exact,d2Gdvpa2_M_num,d2Gdvpa2_M_err,"d2Gdvpa2_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
            dGdvperp_M_max, dGdvperp_M_L2 = print_test_data(dGdvperp_M_exact,dGdvperp_M_num,dGdvperp_M_err,"dGdvperp_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
            d2Gdvperpdvpa_M_max, d2Gdvperpdvpa_M_L2 = print_test_data(d2Gdvperpdvpa_M_exact,d2Gdvperpdvpa_M_num,d2Gdvperpdvpa_M_err,"d2Gdvperpdvpa_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
            d2Gdvperp2_M_max, d2Gdvperp2_M_L2 = print_test_data(d2Gdvperp2_M_exact,d2Gdvperp2_M_num,d2Gdvperp2_M_err,"d2Gdvperp2_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
            atol_max = 2.1e-4
            atol_L2 = 6.5e-6
            @test H_M_max < atol_max
            @test H_M_L2 < atol_L2
            atol_max = 1.5e-3
            atol_L2 = 6.5e-5
            @test dHdvpa_M_max < atol_max
            @test dHdvpa_M_L2 < atol_L2
            atol_max = 8.0e-4
            atol_L2 = 4.0e-5
            @test dHdvperp_M_max < atol_max
            @test dHdvperp_M_L2 < atol_L2
            atol_max = 1.1e-4
            atol_L2 = 4.0e-5
            @test G_M_max < atol_max
            @test G_M_L2 < atol_L2
            atol_max = 2.5e-4
            atol_L2 = 1.2e-5
            @test d2Gdvpa2_M_max < atol_max
            @test d2Gdvpa2_M_L2 < atol_L2
            atol_max = 9.0e-5
            atol_L2 = 6.0e-5
            @test dGdvperp_M_max < atol_max
            @test dGdvperp_M_L2 < atol_L2
            atol_max = 1.1e-4
            atol_L2 = 9.0e-6
            @test d2Gdvperpdvpa_M_max < atol_max
            @test d2Gdvperpdvpa_M_L2 < atol_L2
            atol_max = 2.0e-4
            atol_L2 = 1.1e-5
            @test d2Gdvperp2_M_max < atol_max
            @test d2Gdvperp2_M_L2 < atol_L2
        end
        
        @testset "backward-Euler linearised test particle collisions" begin
            println("    - test backward-Euler linearised test particle collisions")
            @testset "$bc" for bc in (natural_boundary_condition, zero_boundary_condition)  
                println("        -  bc=$bc")
                backward_Euler_linearised_collisions_test(bc_vpa=bc,bc_vperp=bc,
                 use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=true)
                backward_Euler_linearised_collisions_test(bc_vpa=bc,bc_vperp=bc,
                 use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=false,
                 atol_vth=3.0e-7)
            end
        end

        @testset "numerical error correcting terms" begin
            println("    - test numerical error correcting terms")
            numerical_error_corrections_test(print_to_screen=print_to_screen)
        end
        
        
    end
end

end #FokkerPlanckTestsBase

using .FokkerPlanckTestsBase

FokkerPlanckTestsBase.runtests()

