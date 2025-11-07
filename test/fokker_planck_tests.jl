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
using FokkerPlanck.fokker_planck_calculus: direct_integration, multipole_expansion, delta_f_multipole, boundary_data_type,
                                            repeat_assembly_per_species, single_assembly_per_species, multi_species_operator_type

using FokkerPlanck: fokker_planck_backward_euler_data, fokker_planck_collision_operator_weak_form!
using FokkerPlanck: conserving_corrections!, species_info, fixed_background_plasma_input
using FokkerPlanck: fokker_planck_collisions_backward_euler_step!, calculate_entropy_production
using FokkerPlanck.fokker_planck_test: print_test_data, fkpl_error_data, allocate_error_data #, plot_test_data
using FokkerPlanck.fokker_planck_test: F_Maxwellian, G_Maxwellian, H_Maxwellian, F_Beam
using FokkerPlanck.fokker_planck_test: d2Gdvpa2_Maxwellian, d2Gdvperp2_Maxwellian, d2Gdvperpdvpa_Maxwellian, dGdvperp_Maxwellian
using FokkerPlanck.fokker_planck_test: dHdvperp_Maxwellian, dHdvpa_Maxwellian, Cssp_Maxwellian_inputs
using FokkerPlanck.fokker_planck_calculus: calculate_rosenbluth_potentials_via_elliptic_solve!, calculate_rosenbluth_potential_boundary_data_exact!
using FokkerPlanck.fokker_planck_calculus: test_rosenbluth_potential_boundary_data, rosenbluth_potential_boundary_data
using FokkerPlanck.fokker_planck_calculus: enforce_vpavperp_BCs!, calculate_rosenbluth_potentials_via_direct_integration!
using FokkerPlanck.fokker_planck_calculus: interpolate_2D_vspace!, calculate_test_particle_preconditioner!
using FokkerPlanck.fokker_planck_calculus: advance_linearised_test_particle_collisions!, fokkerplanck_weakform_arrays_struct,
                                            fokkerplanck_arrays_direct_integration_struct, calculate_rosenbluth_potentials_via_analytical_Maxwellian!,
                                            convert_rosenbluth_potentials_from_source_to_other_grid!, rosenbluth_potential_data,
                                            calculate_analytical_Maxwellian_multipole_expansion_moments!, delta_f_multipole_moments

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
    species = species_info([ms],[1.0])
    fkpl_arrays = fokker_planck_backward_euler_data(vpa,vperp,species,boundary_data_option,repeat_assembly_per_species,
                        0.0, 0.0, 0, print_to_screen, nothing, nothing)
    dummy_array = allocate_float(vpa.n,vperp.n)
    FMaxwell = allocate_float(vpa.n,vperp.n)
    FMaxwell_err = allocate_float(vpa.n,vperp.n)
    # make sure to use anyv communicator for any array that is modified in fokker_planck.jl functions
    pdf = allocate_float(vpa.n,vperp.n)

    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                FMaxwell[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
                pdf[ivpa,ivperp] = F_Beam(vpa0,vperp0,vth0,vpa.grid[ivpa],vperp.grid[ivperp])
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
    pressure = get_pressure(pdf, vpa, vperp, upar, mass)
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
    pressure = get_pressure(pdf, vpa, vperp, upar, mass)
    vth = sqrt(2.0*pressure/(dens*mass))
    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                pdf_exact[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
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
    multi_species_operator_option=repeat_assembly_per_species,
    print_to_screen=true,
    # error tolerances
    atol_max = 2.0e-5,
    atol_L2 = 2.0e-6,
    atol_dens = 1.0e-8,
    atol_upar = 5.0e-9,
    atol_vth = 1.0e-7)

    vpa, vperp = create_grids(ngrid,nelement_vpa,nelement_vperp;
                      Lvpa=Lvpa,Lvperp=Lvperp,bc_vpa=bc_vpa,bc_vperp=bc_vperp)
    species = species_info([1.0],[1.0])
    nl_solver_atol=1.0e-10
    nl_solver_rtol=0.0
    nl_solver_nonlinear_max_iterations=20
    fkpl_arrays = fokker_planck_backward_euler_data(vpa,vperp,species,boundary_data_option,multi_species_operator_option,
                        nl_solver_atol,nl_solver_rtol,nl_solver_nonlinear_max_iterations,
                        print_to_screen,nothing,nothing)

    # initial condition
    Fold = allocate_float(vpa.n,vperp.n,species.n)
    @inbounds begin
        for is in 1:species.n
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    Fold[ivpa,ivperp,is] = F_Beam(vpa0,vperp0,vth0,vpa.grid[ivpa],vperp.grid[ivperp])
                end
            end
        end
    end
    if vpa.bc == zero_boundary_condition
        @inbounds begin
            for is in 1:species.n
                for ivperp in 1:vperp.n
                    Fold[1,ivperp,is] = 0.0
                    Fold[end,ivperp,is] = 0.0
                end
            end
        end
    end
    if vperp.bc == zero_boundary_condition
        @inbounds begin
            for is in 1:species.n
                for ivpa in 1:vpa.n
                    Fold[ivpa,end,is] = 0.0
                end
            end
        end
    end
    # normalise to unit density
    @inbounds begin
        for is in 1:species.n
        @views densfac = get_density(Fold[:,:,is],vpa,vperp)
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    Fold[ivpa,ivperp,is] /= densfac
                end
            end
        end
    end
    # dummy arrays
    Fdummy1 = allocate_float(vpa.n,vperp.n)
    Fdummy2 = allocate_float(vpa.n,vperp.n)
    Fdummy3 = allocate_float(vpa.n,vperp.n)
    FMaxwell = allocate_float(vpa.n,vperp.n,species.n)
    density = allocate_float(species.n)
    upar = allocate_float(species.n)
    vth = allocate_float(species.n)
    # physics parameters
    nuss = 1.0
    # initial condition
    time = 0.0
    # Maxwellian and parameters
    @inbounds begin
        for is in 1:species.n
            @views density[is] = get_density(Fold[:,:,is],vpa,vperp)
            @views upar[is] = get_upar(Fold[:,:,is], vpa, vperp, density[is])
            @views pressure = get_pressure(Fold[:,:,is], vpa, vperp, upar[is], species.mass[is])
            vth[is] = sqrt(2.0*pressure/(density[is]*species.mass[is]))
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    FMaxwell[ivpa,ivperp,is] = F_Maxwellian(density[is],upar[is],vth[is],vpa.grid[ivpa],vperp.grid[ivperp])
                end
            end
        end
    end

    if print_to_screen
        for is in 1:species.n
            @views diagnose_F_Maxwellian(Fold[:,:,is],Fdummy1,Fdummy2,Fdummy3,vpa,vperp,time,species.mass[is],0)
        end
    end
    for it in 1:ntime
        fokker_planck_collisions_backward_euler_step!(Fold, delta_t, nuss, fkpl_arrays,
            use_conserving_corrections=test_numerical_conserving_terms,
            test_particle_preconditioner=test_particle_preconditioner,
            test_linearised_advance=test_linearised_advance,
            use_Maxwellian_Rosenbluth_coefficients_in_preconditioner=use_Maxwellian_Rosenbluth_coefficients_in_preconditioner)
        # update the pdf
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
        # diagnose Fold
        time += delta_t
        if print_to_screen
            for is in 1:species.n
                @views diagnose_F_Maxwellian(Fold[:,:,is],Fdummy1,Fdummy2,Fdummy3,vpa,vperp,time,species.mass[is],0)
            end
        end
    end

    # now check distribution
    for is in 1:species.n
        @views test_F_Maxwellian(FMaxwell[:,:,is],Fold[:,:,is],
                    vpa,vperp,
                    Fdummy2,Fdummy3,
                    density[is], upar[is], vth[is], species.mass[is],
                    atol_max, atol_L2,
                    atol_dens, atol_upar, atol_vth,
                    print_to_screen=print_to_screen)
    end

    return nothing
end

function get_total_parallel_momentum(ff::AbstractArray{mk_float,3},
    vpa::finite_element_coordinate,
    vperp::finite_element_coordinate,
    species::species_info)
    parallel_momentum = 0.0
    for is in 1:species.n
        @views gamma = species.n0ref[is]*(species.c0ref[is]*get_upar(ff[:,:,is],vpa,vperp,1.0)
                                 + species.u0ref[is]*get_density(ff[:,:,is],vpa,vperp))
        parallel_momentum += species.mass[is]*gamma
    end
    return parallel_momentum
end

function get_total_energy(ff::AbstractArray{mk_float,3},
    vpa::finite_element_coordinate,
    vperp::finite_element_coordinate,
    species::species_info)
    energy = 0.0
    for is in 1:species.n
        @views energy += ((species.n0ref[is]*species.c0ref[is]^2)*
                            get_pressure(ff[:,:,is],vpa,vperp,
                                -species.u0ref[is]/species.c0ref[is],species.mass[is]))
    end
    return energy
end

function multi_species_numerical_error_corrections_test(;
    ngrid = 5,
    nelement_vpa = 8,
    nelement_vperp = 4,
    Lvpa = 12.0,
    Lvperp = 6.0,
    abeam = 0.5,
    vpa0 = 1.0,
    vperp0 = 1.0,
    vth0 = 0.5,
    atol = 5.0e-14,
    print_to_screen=false,
    )
    vpa, vperp = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                                                Lvpa=Lvpa,Lvperp=Lvperp)
    boundary_data_option = multipole_expansion
    species = species_info([1.0,2.0],[1.0,2.0])
    fkpl_arrays = fokkerplanck_weakform_arrays_struct(vpa,vperp,species,boundary_data_option,
                                        print_to_screen=print_to_screen)

    pdf_new = allocate_float(vpa.n,vperp.n,species.n)
    pdf_old = allocate_float(vpa.n,vperp.n,species.n)
    # initialise a distribution that has a qpar
    for is in 1:species.n
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                pdf_new[ivpa,ivperp,is] = (abeam * F_Beam(vpa0,vperp0,vth0,vpa.grid[ivpa],vperp.grid[ivperp])
                                            + F_Beam(0.0,vperp0,vth0,vpa.grid[ivpa],vperp.grid[ivperp]))
            end
        end
    end
    for is in 1:species.n
        mass = species.mass
        @views density = get_density(pdf_new[:,:,is], vpa, vperp)
        @views upar = get_upar(pdf_new[:,:,is], vpa, vperp, density)
        @views pressure = get_pressure(pdf_new[:,:,is], vpa, vperp, upar, mass[is])
        @views ppar = get_ppar(pdf_new[:,:,is], vpa, vperp, upar, mass[is])
        @views qpar = get_qpar(pdf_new[:,:,is], vpa, vperp, upar, mass[is])
        @views rmom = get_rmom(pdf_new[:,:,is], vpa, vperp, upar, mass[is])
        # println("density: $density")
        # println("upar: $upar")
        # println("pressure: $pressure")
        # println("ppar: $ppar")
        # println("qpar: $qpar")
        # println("rmom: $rmom")
        # println("qpar/ppar $(qpar/ppar)")
        # check test pdf unchanged, and has nonzero qpar
        if abeam == 0.5 && vpa0 == 1.0 && vperp0 == 1.0 && vth0 == 0.5
            @test isapprox(density, 7.416900452984803, atol=atol)
            @test isapprox(upar, 0.33114644602432997, atol=atol)
            @test isapprox(pressure, mass[is]*4.242094519010763, atol=atol)
            @test isapprox(ppar, mass[is]*2.5479896423369506, atol=atol)
            @test isapprox(qpar, mass[is]*0.29147880412034594, atol=atol)
            @test isapprox(rmom, mass[is]*27.57985752143237, atol=6*atol)
        end
    end


    densitys, upars, vths = [1.1, 0.9], [1.0, 0.75], [1.0, 1.0]
    @inbounds begin
        for is in 1:species.n
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    pdf_old[ivpa,ivperp,is] = F_Maxwellian(densitys[is],upars[is],vths[is],vpa.grid[ivpa],vperp.grid[ivperp])
                end
            end
        end
    end

    # make ad-hoc conserving corrections to make the density, total momentum and total energy
    # of pdf_new equal to those in pdf_old
    conserving_corrections!(pdf_new,pdf_old,fkpl_arrays,nothing)

    # check pdf_new and pdf_old now have the same density, total momentum and total energy moments
    for is in 1:species.n
        @views n_new = get_density(pdf_new[:,:,is], vpa, vperp)
        @views n_old = get_density(pdf_old[:,:,is], vpa, vperp)
        @test abs(n_new-n_old) < atol
    end
    # compute total parallel momentum
    parallel_momentum_new = get_total_parallel_momentum(pdf_new,vpa,vperp,species)
    parallel_momentum_old = get_total_parallel_momentum(pdf_old,vpa,vperp,species)
    @test abs(parallel_momentum_new - parallel_momentum_old) < atol
    # compute total energy
    energy_new = get_total_energy(pdf_new,vpa,vperp,species)
    energy_old = get_total_energy(pdf_old,vpa,vperp,species)
    @test abs(energy_new - energy_old) < atol
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
            Fe[ivpa,ivperp] = F_Maxwellian(dense,upare,vthe,vpa.grid[ivpa],vperp.grid[ivperp])
            Fe_exact_ion_units[ivpa,ivperp] = F_Maxwellian(dense,upare/scalefac,vthe/scalefac,vpa.grid[ivpa],vperp.grid[ivperp])/(scalefac^3)
            Fi[ivpa,ivperp] = F_Maxwellian(densi,upari,vthi,vpa.grid[ivpa],vperp.grid[ivperp])
            Fi_exact_electron_units[ivpa,ivperp] = (scalefac^3)*F_Maxwellian(densi,upari*scalefac,vthi*scalefac,vpa.grid[ivpa],vperp.grid[ivperp])
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

function test_rosenbluth_potential_grid_conversion(; ngrid=9,
                                nelement_vpa=16,
                                nelement_vperp = 8,
                                Lvpa=12.0, Lvperp=6.0,
                                rtol = 5.0e-6,
                                atol = 1.0e-14,
                                boundary_data_option=multipole_expansion::boundary_data_type,
                                print_to_screen=false)
    vpa, vperp = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                Lvpa=Lvpa,Lvperp=Lvperp)
    electron_mass = 1.0/1836.0
    thermal_temperature = 1.0
    mass = [electron_mass, 1.0, 16.0, 16.0, 8.0]
    zeds = [-1.0, 1.0, 8.0, 8.0, 4.0]
    # reference speeds in units of cref = sqrt(Tref/mref)
    c0ref = [sqrt(2.0*thermal_temperature/mass[is]) for is in 1:length(mass)]
    u0ref = [-0.5, 1.0, 20.0, -30.0, 1.5]
    # reference density in units of nref
    n0ref = [0.9, 1.1, 1.5, 1.4, 0.7]
    species = species_info(mass,zeds,c0ref,u0ref,n0ref)
    # moments of pdfs in units of cref, nref
    density = [0.6, 1.5, 0.7, 0.9, 0.8]
    upar = [-0.6, 1.2, 20.1, -30.1, 1.3]
    vth = [0.8*c0ref[1], 1.2*c0ref[2], 1.1*c0ref[3], 1.05*c0ref[4], 0.9*c0ref[5]]

    # Rosenbluth potentials on natural grids
    rosenbluth_potentials_s = Vector{rosenbluth_potential_data}(undef,species.n)
    # Rosenbluth potentials on grid of another species, analytical
    rosenbluth_potentials_s_converted_exact = rosenbluth_potential_data(vpa,vperp,boundary_data_option)
    # Rosenbluth potentials converted from natural grids to grid of another species
    rosenbluth_potentials_s_converted_numerical = rosenbluth_potential_data(vpa,vperp,boundary_data_option)
    # array for testing errors
    vpavperp_err = allocate_float(vpa.n,vperp.n)
    for is in 1:species.n
        rosenbluth_potentials_s[is] = rosenbluth_potential_data(vpa,vperp,boundary_data_option)
        # get moments for Rosenbluth potentials on natural grids
        density_in = density[is]/n0ref[is]
        upar_in = (upar[is] - u0ref[is])/c0ref[is]
        vth_in = vth[is]/c0ref[is]
        calculate_rosenbluth_potentials_via_analytical_Maxwellian!(
            rosenbluth_potentials_s[is],density_in,upar_in,vth_in,vpa,vperp)
    end
    for is in 1:species.n
        # test the conversion of species s to s' grids for each primed species
        for isp in 1:species.n
            # first compute exact results for species s on s' grid
            density_in = density[is]/n0ref[isp]
            upar_in = (upar[is] - u0ref[isp])/c0ref[isp]
            vth_in = vth[is]/c0ref[isp]
            calculate_rosenbluth_potentials_via_analytical_Maxwellian!(
                rosenbluth_potentials_s_converted_exact,density_in,upar_in,vth_in,vpa,vperp)
            # make normalisation prefactor still units of species s
            @. rosenbluth_potentials_s_converted_exact.GG *= (n0ref[isp]*c0ref[isp])/(n0ref[is]*c0ref[is])
            @. rosenbluth_potentials_s_converted_exact.dGdvperp *= (n0ref[isp]/n0ref[is])
            @. rosenbluth_potentials_s_converted_exact.d2Gdvperp2 *= (n0ref[isp]/c0ref[isp])/(n0ref[is]/c0ref[is])
            @. rosenbluth_potentials_s_converted_exact.d2Gdvperpdvpa *= (n0ref[isp]/c0ref[isp])/(n0ref[is]/c0ref[is])
            @. rosenbluth_potentials_s_converted_exact.d2Gdvpa2 *= (n0ref[isp]/c0ref[isp])/(n0ref[is]/c0ref[is])
            @. rosenbluth_potentials_s_converted_exact.HH *= (n0ref[isp]/c0ref[isp])/(n0ref[is]/c0ref[is])
            @. rosenbluth_potentials_s_converted_exact.dHdvpa *= (n0ref[isp]/c0ref[isp]^2)/(n0ref[is]/c0ref[is]^2)
            @. rosenbluth_potentials_s_converted_exact.dHdvperp *= (n0ref[isp]/c0ref[isp]^2)/(n0ref[is]/c0ref[is]^2)
            # do the grid interpolation/extrapolation
            convert_rosenbluth_potentials_from_source_to_other_grid!(rosenbluth_potentials_s_converted_numerical,
                rosenbluth_potentials_s[is], vpa, vperp, species.c0ref[is], species.u0ref[is],
                species.c0ref[isp], species.u0ref[isp],calculate_GG=true,calculate_dGdvperp=true)
            # test G
            @. vpavperp_err = abs(rosenbluth_potentials_s_converted_numerical.GG - rosenbluth_potentials_s_converted_exact.GG)
            max_G_err = maximum(vpavperp_err)
            max_G = maximum(abs.(rosenbluth_potentials_s_converted_exact.GG))
            # test dGdvperp
            @. vpavperp_err = abs(rosenbluth_potentials_s_converted_numerical.dGdvperp - rosenbluth_potentials_s_converted_exact.dGdvperp)
            max_dGdvperp_err = maximum(vpavperp_err)
            max_dGdvperp = maximum(abs.(rosenbluth_potentials_s_converted_exact.dGdvperp))
            # test d2Gdvperp2
            @. vpavperp_err = abs(rosenbluth_potentials_s_converted_numerical.d2Gdvperp2 - rosenbluth_potentials_s_converted_exact.d2Gdvperp2)
            max_d2Gdvperp2_err = maximum(vpavperp_err)
            max_d2Gdvperp2 = maximum(abs.(rosenbluth_potentials_s_converted_exact.d2Gdvperp2))
            # test d2Gdvperpdvpa
            @. vpavperp_err = abs(rosenbluth_potentials_s_converted_numerical.d2Gdvperpdvpa - rosenbluth_potentials_s_converted_exact.d2Gdvperpdvpa)
            max_d2Gdvperpdvpa_err = maximum(vpavperp_err)
            max_d2Gdvperpdvpa = maximum(abs.(rosenbluth_potentials_s_converted_exact.d2Gdvperpdvpa))
            # test d2Gdvpa2
            @. vpavperp_err = abs(rosenbluth_potentials_s_converted_numerical.d2Gdvpa2 - rosenbluth_potentials_s_converted_exact.d2Gdvpa2)
            max_d2Gdvpa2_err = maximum(vpavperp_err)
            max_d2Gdvpa2 = maximum(abs.(rosenbluth_potentials_s_converted_exact.d2Gdvpa2))
            # test H
            @. vpavperp_err = abs(rosenbluth_potentials_s_converted_numerical.HH - rosenbluth_potentials_s_converted_exact.HH)
            max_H_err = maximum(vpavperp_err)
            max_H = maximum(abs.(rosenbluth_potentials_s_converted_exact.HH))
            # test dHdvpa
            @. vpavperp_err = abs(rosenbluth_potentials_s_converted_numerical.dHdvpa - rosenbluth_potentials_s_converted_exact.dHdvpa)
            max_dHdvpa_err = maximum(vpavperp_err)
            max_dHdvpa = maximum(abs.(rosenbluth_potentials_s_converted_exact.dHdvpa))
            # test dHdvperp
            @. vpavperp_err = abs(rosenbluth_potentials_s_converted_numerical.dHdvperp - rosenbluth_potentials_s_converted_exact.dHdvperp)
            max_dHdvperp_err = maximum(vpavperp_err)
            max_dHdvperp = maximum(abs.(rosenbluth_potentials_s_converted_exact.dHdvperp))
            if print_to_screen
                println("is=$is isp=$isp")
                println("max_G=$max_G max_G_err=$max_G_err")
                println("max_dGdvperp=$max_dGdvperp max_dGdvperp_err=$max_dGdvperp_err")
                println("max_d2Gdvperp2=$max_d2Gdvperp2 max_d2Gdvperp2_err=$max_d2Gdvperp2_err")
                println("max_d2Gdvperpdvpa=$max_d2Gdvperpdvpa max_d2Gdvperpdvpa_err=$max_d2Gdvperpdvpa_err")
                println("max_d2Gdvpa2=$max_d2Gdvpa2 max_d2Gdvpa2_err=$max_d2Gdvpa2_err")
                println("max_H=$max_H max_H_err=$max_H_err")
                println("max_dHdvpa=$max_dHdvpa max_dHdvpa_err=$max_dHdvpa_err")
                println("max_dHdvperp=$max_dHdvperp max_dHdvperp_err=$max_dHdvperp_err")
            end
            @test max_G_err < rtol * max_G + atol
            @test max_dGdvperp_err < rtol * max_dGdvperp + atol
            @test max_d2Gdvperp2_err < rtol * max_d2Gdvperp2 + atol
            @test max_d2Gdvperpdvpa_err < rtol * max_d2Gdvperpdvpa + atol
            @test max_d2Gdvpa2_err < rtol * max_d2Gdvpa2 + atol
            @test max_H_err < rtol * max_H + atol
            @test max_dHdvpa_err < rtol * max_dHdvpa + atol
            @test max_dHdvperp_err < rtol * max_dHdvperp + atol
        end
    end
    return nothing
end

function multi_species_fokker_planck_collisions_test(; ngrid=17, nelement_vpa=8, nelement_vperp=4,
                                    # set small absolute values for test tolerances
                                    atol_max = 1.0e-5,
                                    atol_L2 = 1.0e-7,
                                    print_to_screen=false
                                    )
    nuref = 1.0
    #test_numerical_conserving_terms = false
    test_Maxwellian_Rosenbluth_coefficients = false
    density = [1.0, 1.0, 1.0]
    upar = [1.0, -0.7, 0.2]
    vth = [1.0,1.0,1.0]
    @testset "boundary_data_option=$boundary_data_option mass=$(species.mass) zeds=$(species.zeds) bc=$(bc) multi_species_operator_option=$(multi_species_operator_option)" for
            (boundary_data_option, species, bc, multi_species_operator_option) in (#(direct_integration,species_info([0.5],[2.0]),),
                                                (multipole_expansion,species_info([0.5],[2.0]),natural_boundary_condition,single_assembly_per_species),
                                                (multipole_expansion,species_info([0.5],[2.0]),zero_boundary_condition,repeat_assembly_per_species),
                                                (delta_f_multipole,species_info([0.5],[2.0]),natural_boundary_condition,single_assembly_per_species),
                                                (delta_f_multipole,species_info([0.5,1.0],[2.0,1.0]),natural_boundary_condition,single_assembly_per_species),
                                                (delta_f_multipole,species_info([0.5,1.0],[2.0,1.0]),zero_boundary_condition,repeat_assembly_per_species),
                                                (delta_f_multipole,species_info([0.5,1.0,2.0],[2.0,-1.0,1.0]),natural_boundary_condition,single_assembly_per_species),
                                                )
        vpa, vperp = create_grids(ngrid,nelement_vpa,nelement_vperp,
            Lvpa=10.0,Lvperp=5.0,bc_vpa=bc,bc_vperp=bc)
        println("       - boundary_data_option=$boundary_data_option mass=$(species.mass) zeds=$(species.zeds) bc=$(bc) multi_species_operator_option=$(multi_species_operator_option)")
        @testset "test_numerical_conserving_terms=$test_numerical_conserving_terms" for
            (test_numerical_conserving_terms,) in (false,true)
            println("           - test_numerical_conserving_terms=$test_numerical_conserving_terms")
            fkpl_arrays = fokkerplanck_weakform_arrays_struct(vpa,vperp,species,boundary_data_option,
                                                            multi_species_operator_option=multi_species_operator_option,
                                                            print_to_screen=print_to_screen)
            # arrays for the test
            F_M = allocate_float(vpa.n,vperp.n,species.n)
            C_M_num = allocate_float(vpa.n,vperp.n,species.n)
            C_M_exact = allocate_float(vpa.n,vperp.n,species.n)
            C_M_err = allocate_float(vpa.n,vperp.n)
            dummy_array = allocate_float(vpa.n,vperp.n)
            mass = species.mass
            zed = species.zeds
            @. C_M_exact = 0.0
            nfac = 0.3
            ufac = 0.7
            # specify a pdf that has a nonzero qpar~ 0.1 pressure by summing Maxwellian distributions
            # F_s = F_sA + nfac * F_sB
            for is in 1:species.n
                for ivperp in 1:vperp.n
                    for ivpa in 1:vpa.n
                        F_M[ivpa,ivperp,is] = (F_Maxwellian(density[is],upar[is],vth[is],vpa.grid[ivpa],vperp.grid[ivperp]) +
                                                nfac*F_Maxwellian(density[is],upar[is]*ufac,vth[is]*ufac,vpa.grid[ivpa],vperp.grid[ivperp]))
                    end
                end
                # use this commented code to assess how far from Maxwellian F_M is
                # @views density_M = get_density(F_M[:,:,is], vpa, vperp)
                # @views upar_M = get_upar(F_M[:,:,is], vpa, vperp, density_M)
                # @views pressure_M = get_pressure(F_M[:,:,is], vpa, vperp, upar_M, mass[is])
                # @views ppar_M = get_ppar(F_M[:,:,is], vpa, vperp, upar_M, mass[is])
                # @views qpar_M = get_qpar(F_M[:,:,is], vpa, vperp, upar_M, mass[is])
                # @views rmom_M = get_rmom(F_M[:,:,is], vpa, vperp, upar_M, mass[is])
                # println("density_M: $density_M")
                # println("upar_M: $upar_M")
                # println("pressure_M: $pressure_M")
                # println("ppar_M: $ppar_M")
                # println("qpar_M: $qpar_M")
                # println("rmom_M: $rmom_M")
                # println("qpar_M/ppar_M $(qpar_M/ppar_M)")
            end
            # sum up contributions to cross-collision operator
            for is in 1:species.n
                for isp in 1:species.n
                    for ivperp in 1:vperp.n
                        for ivpa in 1:vpa.n
                            # obtain an exact expression for the non-Maxwellian pdf
                            # by using that the collision operator is bilinear, i.e.,
                            # C[F_s,F_s'] =  C[F_sA,F_s'A]
                            #                 + nfac * ( C[F_sA,F_s'B] + C[F_sB,F_s'A])
                            #                 + nfac^2 * C[F_sB,F_s'B]
                            C_M_exact[ivpa,ivperp,is] += (Cssp_Maxwellian_inputs(density[is],upar[is],vth[is],mass[is],zed[is],
                                                                            density[isp],upar[isp],vth[isp],mass[isp],zed[isp],
                                                                            nuref,vpa.grid[ivpa],vperp.grid[ivperp]) +
                                                            nfac*Cssp_Maxwellian_inputs(density[is],upar[is]*ufac,vth[is]*ufac,mass[is],zed[is],
                                                                            density[isp],upar[isp],vth[isp],mass[isp],zed[isp],
                                                                            nuref,vpa.grid[ivpa],vperp.grid[ivperp]) +
                                                            nfac*Cssp_Maxwellian_inputs(density[is],upar[is],vth[is],mass[is],zed[is],
                                                                            density[isp],upar[isp]*ufac,vth[isp]*ufac,mass[isp],zed[isp],
                                                                            nuref,vpa.grid[ivpa],vperp.grid[ivperp]) +
                                                            (nfac^2)*Cssp_Maxwellian_inputs(density[is],upar[is]*ufac,vth[is]*ufac,mass[is],zed[is],
                                                                            density[isp],upar[isp]*ufac,vth[isp]*ufac,mass[isp],zed[isp],
                                                                            nuref,vpa.grid[ivpa],vperp.grid[ivperp]))
                        end
                    end
                end
            end
            fokker_planck_collision_operator_weak_form!(C_M_num,
                    F_M, nuref, fkpl_arrays;
                    use_conserving_corrections=test_numerical_conserving_terms,
                    use_Maxwellian_Rosenbluth_coefficients=test_Maxwellian_Rosenbluth_coefficients)
            # test relative values as C_M_exact /= 0 in general
            rtol_max = atol_max
            rtol_L2 = atol_L2
            for is in 1:species.n
                Cnorm = maximum(abs.(@view C_M_exact[:,:,is]))
                @views C_M_max, C_M_L2 = print_test_data(C_M_exact[:,:,is],C_M_num[:,:,is],C_M_err,"C_M[$(is)]",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                #println(Cnorm, " ", C_M_max, " ", C_M_L2)
                @test C_M_max < atol_max + rtol_max*Cnorm
                @test C_M_L2 < atol_L2 + rtol_L2*Cnorm
            end
            # test conservation properties
            if test_numerical_conserving_terms
                atol_n = 5.0e-12
                atol_momentum = 3.0e-12
                atol_energy = 3.0e-12
            else
                atol_n = 1.0e-11
                atol_momentum = 1.0e-8
                atol_energy = 1.0e-8
            end
            # compute changes in density induced by C_M_num
            for is in 1:species.n
                @views delta_n = get_density(C_M_num[:,:,is],vpa,vperp)
                @test delta_n < atol_n
            end
            # compute change in total parallel momentum
            delta_parallel_momentum = get_total_parallel_momentum(C_M_num,vpa,vperp,species)
            @test delta_parallel_momentum < atol_momentum
            # compute change in total energy
            delta_energy = get_total_energy(C_M_num,vpa,vperp,species)
            @test delta_energy < atol_energy
            # check entropy production is positive for pdf far from Maxwellian
            # n.b. dSdt may be negative and small if pdf is close to Maxwellian
            dSdt = calculate_entropy_production(C_M_num,F_M,fkpl_arrays)
            @test dSdt > 0.0
        end
    end
    return nothing
end

function multi_species_multi_reference_fokker_planck_collisions_test(;
                ngrid=17, nelement_vpa=8, nelement_vperp=4, Lvpa=10.0, Lvperp=5.0,
                # set small absolute values for test tolerances
                atol_max = 1.0e-5,
                atol_L2 = 1.0e-7,
                print_to_screen=false,
                #assembly_option=repeat_assembly_per_species::multi_species_operator_type,
                )
    nuref = 1.0
    #test_numerical_conserving_terms = false
    test_Maxwellian_Rosenbluth_coefficients = false
    density = [1.0, 1.0, 1.0]
    upar = [1.0, -0.7, 0.2]
    vth = [2.0,1.0,1.0]
    mass2species = [0.5,1.0]
    zeds2species = [2.0,1.0]
    c0ref2species = [1.9,1.1]
    u0ref2species = [1.1,-0.8]
    n0ref2species = [0.9,1.5]
    @testset "boundary_data_option=$boundary_data_option mass=$(species.mass) zeds=$(species.zeds) bc=$(bc) multi_species_operator_option=$(multi_species_operator_option)" for
            (boundary_data_option, species, bc, multi_species_operator_option) in (
                                                (multipole_expansion,species_info(mass2species,zeds2species,c0ref2species,u0ref2species,n0ref2species),natural_boundary_condition,single_assembly_per_species),
                                                (multipole_expansion,species_info(mass2species,zeds2species,c0ref2species,u0ref2species,n0ref2species),natural_boundary_condition,repeat_assembly_per_species),
                                                )
        vpa, vperp = create_grids(ngrid,nelement_vpa,nelement_vperp,
            Lvpa=Lvpa,Lvperp=Lvperp,bc_vpa=bc,bc_vperp=bc)
        println("       - boundary_data_option=$boundary_data_option mass=$(species.mass) zeds=$(species.zeds) bc=$(bc) multi_species_operator_option=$(multi_species_operator_option)")
        @testset "test_numerical_conserving_terms=$test_numerical_conserving_terms" for
            (test_numerical_conserving_terms,) in (false,true)
            println("           - test_numerical_conserving_terms=$test_numerical_conserving_terms")
            fkpl_arrays = fokkerplanck_weakform_arrays_struct(vpa,vperp,species,boundary_data_option,
                                                            multi_species_operator_option=multi_species_operator_option,
                                                            print_to_screen=print_to_screen)
            # arrays for the test
            F_M = allocate_float(vpa.n,vperp.n,species.n)
            C_M_num = allocate_float(vpa.n,vperp.n,species.n)
            C_M_exact = allocate_float(vpa.n,vperp.n,species.n)
            C_M_err = allocate_float(vpa.n,vperp.n)
            dummy_array = allocate_float(vpa.n,vperp.n)
            mass = species.mass
            zed = species.zeds
            c0ref = species.c0ref
            u0ref = species.u0ref
            n0ref = species.n0ref
            @. C_M_exact = 0.0
            nfac = 0.3
            ufac = 0.7
            # specify a pdf that has a nonzero qpar~ 0.1 pressure by summing Maxwellian distributions
            # F_s = F_sA + nfac * F_sB
            for is in 1:species.n
                prefactor = (c0ref[is]^3)/n0ref[is]
                for ivperp in 1:vperp.n
                    vperp_s = c0ref[is]*vperp.grid[ivperp]
                    for ivpa in 1:vpa.n
                        vpa_s = c0ref[is]*vpa.grid[ivpa] + u0ref[is]
                        F_M[ivpa,ivperp,is] = prefactor*(F_Maxwellian(density[is],upar[is],vth[is],vpa_s,vperp_s) +
                                                nfac*F_Maxwellian(density[is],upar[is]*ufac,vth[is]*ufac,vpa_s,vperp_s))
                    end
                end
                # assess how far from Maxwellian F_M is
                if print_to_screen
                    @views density_M = get_density(F_M[:,:,is], vpa, vperp)
                    @views upar_M = get_upar(F_M[:,:,is], vpa, vperp, density_M)
                    @views pressure_M = get_pressure(F_M[:,:,is], vpa, vperp, upar_M, mass[is])
                    @views ppar_M = get_ppar(F_M[:,:,is], vpa, vperp, upar_M, mass[is])
                    @views qpar_M = get_qpar(F_M[:,:,is], vpa, vperp, upar_M, mass[is])
                    @views rmom_M = get_rmom(F_M[:,:,is], vpa, vperp, upar_M, mass[is])
                    println("density_M: $density_M")
                    println("upar_M: $upar_M")
                    println("pressure_M: $pressure_M")
                    println("ppar_M: $ppar_M")
                    println("qpar_M: $qpar_M")
                    println("rmom_M: $rmom_M")
                    println("qpar_M/ppar_M $(qpar_M/ppar_M)")
                end
            end
            # sum up contributions to cross-collision operator
            for is in 1:species.n
                prefactor = (c0ref[is]^3)/n0ref[is]
                for isp in 1:species.n
                    for ivperp in 1:vperp.n
                        vperp_s = c0ref[is]*vperp.grid[ivperp]
                        for ivpa in 1:vpa.n
                            vpa_s = c0ref[is]*vpa.grid[ivpa] + u0ref[is]
                            # obtain an exact expression for the non-Maxwellian pdf
                            # by using that the collision operator is bilinear, i.e.,
                            # C[F_s,F_s'] =  C[F_sA,F_s'A]
                            #                 + nfac * ( C[F_sA,F_s'B] + C[F_sB,F_s'A])
                            #                 + nfac^2 * C[F_sB,F_s'B]
                            C_M_exact[ivpa,ivperp,is] += prefactor*(Cssp_Maxwellian_inputs(density[is],upar[is],vth[is],mass[is],zed[is],
                                                                            density[isp],upar[isp],vth[isp],mass[isp],zed[isp],
                                                                            nuref,vpa_s,vperp_s) +
                                                            nfac*Cssp_Maxwellian_inputs(density[is],upar[is]*ufac,vth[is]*ufac,mass[is],zed[is],
                                                                            density[isp],upar[isp],vth[isp],mass[isp],zed[isp],
                                                                            nuref,vpa_s,vperp_s) +
                                                            nfac*Cssp_Maxwellian_inputs(density[is],upar[is],vth[is],mass[is],zed[is],
                                                                            density[isp],upar[isp]*ufac,vth[isp]*ufac,mass[isp],zed[isp],
                                                                            nuref,vpa_s,vperp_s) +
                                                            (nfac^2)*Cssp_Maxwellian_inputs(density[is],upar[is]*ufac,vth[is]*ufac,mass[is],zed[is],
                                                                            density[isp],upar[isp]*ufac,vth[isp]*ufac,mass[isp],zed[isp],
                                                                            nuref,vpa_s,vperp_s))
                        end
                    end
                end
            end
            fokker_planck_collision_operator_weak_form!(C_M_num,
                    F_M, nuref, fkpl_arrays;
                    use_conserving_corrections=test_numerical_conserving_terms,
                    use_Maxwellian_Rosenbluth_coefficients=test_Maxwellian_Rosenbluth_coefficients)
            # test relative values as C_M_exact /= 0 in general
            rtol_max = atol_max
            rtol_L2 = atol_L2
            for is in 1:species.n
                Cnorm = maximum(abs.(@view C_M_exact[:,:,is]))
                @views C_M_max, C_M_L2 = print_test_data(C_M_exact[:,:,is],C_M_num[:,:,is],C_M_err,"C_M[$(is)]",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                #println(Cnorm, " ", C_M_max, " ", C_M_L2)
                @test C_M_max < atol_max + rtol_max*Cnorm
                @test C_M_L2 < atol_L2 + rtol_L2*Cnorm
            end
            # test conservation properties
            if test_numerical_conserving_terms
                atol_n = 5.0e-12
                atol_momentum = 3.0e-12
                atol_energy = 3.0e-12
            else
                atol_n = 1.0e-11
                atol_momentum = 1.0e-8
                atol_energy = 1.0e-8
            end
            # compute changes in density induced by C_M_num
            for is in 1:species.n
                @views delta_n = n0ref[is]*get_density(C_M_num[:,:,is],vpa,vperp)
                @test delta_n < atol_n
            end
            # compute change in total parallel momentum
            delta_parallel_momentum = get_total_parallel_momentum(C_M_num,vpa,vperp,species)
            @test delta_parallel_momentum < atol_momentum
            # compute change in total energy
            delta_energy = get_total_energy(C_M_num,vpa,vperp,species)
            @test delta_energy < atol_energy
            # check entropy production is positive for pdf far from Maxwellian
            # n.b. dSdt may be negative and small if pdf is close to Maxwellian
            dSdt = calculate_entropy_production(C_M_num,F_M,fkpl_arrays)
            @test dSdt > 0.0
        end
    end
    return nothing
end

function slowing_down_fokker_planck_collisions_test(;
    ngrid = 9,
    nelement_vpa = 16,
    nelement_vperp = 8,
    bc = natural_boundary_condition,
    multi_species_operator_option = single_assembly_per_species,
    print_to_screen=false)
    vpa, vperp = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                Lvpa=12.0,Lvperp=6.0,bc_vpa=bc,
                                bc_vperp=bc)
    boundary_data_option=multipole_expansion
    species = species_info([1.0], # mass of evolved species
                            [2.0]) # Z of evolved species
    nuref = 1.0/16.0 # reference collision frequency
    # parameters of fixed background species
    msp = [1.0,0.2]#[0.25, 0.25/1836.0]
    Zsp = [1.0,1.0]#[0.5, 0.5]
    denssp = [1.0,1.0]#[1.0, 1.0]
    uparsp = [0.0,0.0]#[0.0, 0.0]
    vthsp = [sqrt(0.5/msp[1]), sqrt(0.5/msp[2])]#[sqrt(0.01/msp[1]), sqrt(0.01/msp[2])]

    @testset "multi_species_operator_option=$multi_species_operator_option bc=$bc pdf_input=$(pdf_input)" for pdf_input in (true,false)
        println("        - multi_species_operator_option=$multi_species_operator_option bc=$bc pdf_input=$pdf_input")
        if pdf_input
            nsprime = length(msp)
            Fsp_M = allocate_float(vpa.n,vperp.n,nsprime)
            for isp in 1:nsprime
                for ivperp in 1:vperp.n
                    for ivpa in 1:vpa.n
                        Fsp_M[ivpa,ivperp,isp] = F_Maxwellian(denssp[isp],uparsp[isp],vthsp[isp],vpa.grid[ivpa],vperp.grid[ivperp])
                    end
                end
            end
            fixed_background_plasma_in = fixed_background_plasma_input(msp,Zsp,Fsp_M)
        else
            fixed_background_plasma_in = fixed_background_plasma_input(msp,Zsp,denssp,uparsp,vthsp)
        end
        fkpl_arrays = fokkerplanck_weakform_arrays_struct(vpa,vperp,species,boundary_data_option,
                                multi_species_operator_option=multi_species_operator_option,
                                print_to_screen=print_to_screen,
                                fixed_background_plasma_in=fixed_background_plasma_in)
        dummy_array = allocate_float(vpa.n,vperp.n)
        Fs_M = allocate_float(vpa.n,vperp.n,species.n)
        C_M_num = allocate_float(vpa.n,vperp.n,species.n)
        C_M_exact = allocate_float(vpa.n,vperp.n,species.n)
        C_M_err = allocate_float(vpa.n,vperp.n)

        # pick a set of parameters that represent slowing down
        # on slow ions and faster electrons, but which are close
        # enough to 1 for errors comparable to the self-collision operator
        # increasing or reducing vth, mass increases the errors
        dens, upar, vth = [1.0], [1.0], [1.0]
        nsprime = size(msp,1)

        for is in 1:species.n
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    Fs_M[ivpa,ivperp,is] = F_Maxwellian(dens[is],upar[is],vth[is],vpa.grid[ivpa],vperp.grid[ivperp])
                    C_M_exact[ivpa,ivperp,is] = 0.0
                end
            end
        end
        for is in 1:species.n
            ms = species.mass[is]
            Zs = species.zeds[is]
            # sum up contributions from evolved species
            for isp in 1:species.n
                for ivperp in 1:vperp.n
                    for ivpa in 1:vpa.n
                            C_M_exact[ivpa,ivperp,is] += Cssp_Maxwellian_inputs(dens[is],upar[is],vth[is],species.mass[is],species.zeds[is],
                                                                        dens[isp],upar[isp],vth[isp],species.mass[isp],species.zeds[isp],
                                                                        nuref,vpa.grid[ivpa],vperp.grid[ivperp])
                    end
                end
            end
            # sum up contributions to cross-collision operator from fixed background
            for isp in 1:nsprime
                for ivperp in 1:vperp.n
                    for ivpa in 1:vpa.n
                            C_M_exact[ivpa,ivperp,is] += Cssp_Maxwellian_inputs(dens[is],upar[is],vth[is],species.mass[is],species.zeds[is],
                                                                        denssp[isp],uparsp[isp],vthsp[isp],msp[isp],Zsp[isp],
                                                                        nuref,vpa.grid[ivpa],vperp.grid[ivperp])
                    end
                end
            end
        end
        @testset "test_numerical_conserving_terms=$test_numerical_conserving_terms" for test_numerical_conserving_terms in (true,false)
            println("           - test_numerical_conserving_terms=$test_numerical_conserving_terms")
            fokker_planck_collision_operator_weak_form!(
                        C_M_num,Fs_M,nuref,fkpl_arrays;
                        use_conserving_corrections=test_numerical_conserving_terms)
            for is in 1:species.n
                @views C_M_max, C_M_L2 = print_test_data(C_M_exact[:,:,is],C_M_num[:,:,is],C_M_err,"C_M",vpa,vperp,dummy_array,print_to_screen=print_to_screen)
                atol_max = 5.0e-6
                atol_L2 = 5.0e-8
                @test C_M_max < atol_max
                @test C_M_L2 < atol_L2
                if !test_numerical_conserving_terms
                    @views delta_n = get_density(C_M_num[:,:,is], vpa, vperp)
                    rtol, atol = 0.0, 1.0e-12
                    @test isapprox(delta_n, rtol ; atol=atol)
                    if print_to_screen
                        println("delta_n: ", delta_n)
                    end
                elseif test_numerical_conserving_terms
                    @views delta_n = get_density(C_M_num[:,:,is], vpa, vperp)
                    rtol, atol = 0.0, 5.0e-14
                    @test isapprox(delta_n, rtol ; atol=atol)
                    if print_to_screen
                        println("delta_n: ", delta_n)
                    end
                end
            end
        end
    end
    return nothing
end

function rosenbluth_potential_solver_test(;
                    ngrid = 9,
                    nelement_vpa = 8,
                    nelement_vperp = 4,
                    boundary_data_option=multipole_expansion,
                    print_to_screen=false)
    vpa, vperp = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                Lvpa=12.0,Lvperp=6.0)
    species = species_info([1.0],[1.0])
    fkpl_arrays = fokkerplanck_weakform_arrays_struct(vpa,vperp,species,boundary_data_option,
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
    Inm_vec = allocate_float(25)
    Inm_vec_exact = allocate_float(25)
    Inm_vec_err = allocate_float(25)

    dens, upar, vth = 0.8, 0.99, 1.01

    for ivperp in 1:vperp.n
        for ivpa in 1:vpa.n
            F_M[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
            H_M_exact[ivpa,ivperp] = H_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
            G_M_exact[ivpa,ivperp] = G_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
            d2Gdvpa2_M_exact[ivpa,ivperp] = d2Gdvpa2_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
            d2Gdvperp2_M_exact[ivpa,ivperp] = d2Gdvperp2_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
            dGdvperp_M_exact[ivpa,ivperp] = dGdvperp_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
            d2Gdvperpdvpa_M_exact[ivpa,ivperp] = d2Gdvperpdvpa_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
            dHdvpa_M_exact[ivpa,ivperp] = dHdvpa_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
            dHdvperp_M_exact[ivpa,ivperp] = dHdvperp_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
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
            fkpl_arrays.rosenbluth_potentials, F_M, vpa, vperp,
            fkpl_arrays.fprp_solver_data; algebraic_solve_for_d2Gdvperp2=false,
            calculate_GG=true, calculate_dGdvperp=true)
    # extract C[Fs,Fs'] result
    # and Rosenbluth potentials for testing


    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
            G_M_num[ivpa,ivperp] = fkpl_arrays.rosenbluth_potentials.GG[ivpa,ivperp]
            H_M_num[ivpa,ivperp] = fkpl_arrays.rosenbluth_potentials.HH[ivpa,ivperp]
            dHdvpa_M_num[ivpa,ivperp] = fkpl_arrays.rosenbluth_potentials.dHdvpa[ivpa,ivperp]
            dHdvperp_M_num[ivpa,ivperp] = fkpl_arrays.rosenbluth_potentials.dHdvperp[ivpa,ivperp]
            dGdvperp_M_num[ivpa,ivperp] = fkpl_arrays.rosenbluth_potentials.dGdvperp[ivpa,ivperp]
            d2Gdvperp2_M_num[ivpa,ivperp] = fkpl_arrays.rosenbluth_potentials.d2Gdvperp2[ivpa,ivperp]
            d2Gdvpa2_M_num[ivpa,ivperp] = fkpl_arrays.rosenbluth_potentials.d2Gdvpa2[ivpa,ivperp]
            d2Gdvperpdvpa_M_num[ivpa,ivperp] = fkpl_arrays.rosenbluth_potentials.d2Gdvperpdvpa[ivpa,ivperp]
            end
        end
    end
    if boundary_data_option in (multipole_expansion,delta_f_multipole)
        if boundary_data_option == multipole_expansion
            calculate_analytical_Maxwellian_multipole_expansion_moments!(Inm_vec_exact,
                                                                    dens,upar,vth)
            Inm_vec .= fkpl_arrays.rosenbluth_potentials.multipole_expansion_moments
        elseif boundary_data_option == delta_f_multipole
            expansion_data_exact = delta_f_multipole_moments(Inm_vec_exact,[0.0,0.0,0.0])
            calculate_analytical_Maxwellian_multipole_expansion_moments!(expansion_data_exact,
                                                                    dens,upar,vth)
            Inm_vec_exact = expansion_data_exact.Inm_vec
            Inm_vec .= fkpl_arrays.rosenbluth_potentials.multipole_expansion_moments.Inm_vec
            @test isapprox(fkpl_arrays.rosenbluth_potentials.multipole_expansion_moments.Maxwellian_moments,
                            expansion_data_exact.Maxwellian_moments,atol=1.0e-9)
        end
        @. Inm_vec_err = abs(Inm_vec - Inm_vec_exact)
        #println(Inm_vec_exact)
        #println(Inm_vec_err./Inm_vec_exact)
        rtol_Inm = 2.0e-8
        atol_Inm = 2.0e-8
        for j in 1:length(Inm_vec)
            @test Inm_vec_err[j] < rtol_Inm*Inm_vec_exact[j] + atol_Inm
        end
    end
    # test the boundary data
    max_H_boundary_data_err, max_dHdvpa_boundary_data_err,
    max_dHdvperp_boundary_data_err, max_G_boundary_data_err,
    max_dGdvperp_boundary_data_err, max_d2Gdvperp2_boundary_data_err,
    max_d2Gdvperpdvpa_boundary_data_err, max_d2Gdvpa2_boundary_data_err = test_rosenbluth_potential_boundary_data(fkpl_arrays.fprp_solver_data.rpbd,rpbd_exact,vpa,vperp,print_to_screen=print_to_screen)
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
        atol_max_G = 2.0e-11
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

        @testset "conversion of Rosenbluth potentials from source to other grids" begin
            println("    - test conversion of Rosenbluth potentials from source to other grids")
            @testset "$boundary_data_option" for boundary_data_option in (multipole_expansion,delta_f_multipole)
                println("        -  boundary_data_option=$boundary_data_option")
                test_rosenbluth_potential_grid_conversion(boundary_data_option=boundary_data_option)
            end
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
            species = species_info([1.0],[1.0])
            fkpl_arrays = fokkerplanck_weakform_arrays_struct(vpa,vperp,species,boundary_data_option,
                                                    print_to_screen=print_to_screen)
            matrix_operators = fkpl_arrays.fprp_solver_data.matrix_operators
            KKpar2D_with_BC_terms_sparse = matrix_operators.KKpar2D_with_BC_terms_sparse
            KKperp2D_with_BC_terms_sparse = matrix_operators.KKperp2D_with_BC_terms_sparse
            lu_obj_MM = matrix_operators.lu_obj_MM

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
                rosenbluth_potential_solver_test(ngrid = 9, nelement_vpa = 8, nelement_vperp = 4,
                    boundary_data_option=boundary_data_option, print_to_screen=print_to_screen)
            end
        end

        @testset "weak-form collision operator calculation" begin
            println("    - test weak-form collision operator calculation")
            ngrid = 9
            nelement_vpa = 8
            nelement_vperp = 4
            vpa, vperp = create_grids(ngrid,nelement_vpa,nelement_vperp,
                                    Lvpa=12.0,Lvperp=6.0,
                                    bc_vpa=natural_boundary_condition,
                                    bc_vperp=natural_boundary_condition)
            boundary_data_option=direct_integration
            species = species_info([1.0],[1.0])
            fkpl_arrays = fokkerplanck_weakform_arrays_struct(vpa,vperp,species,boundary_data_option,
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
                ms = species.mass[1]
                Zs = species.zeds[1]
                msp = 1.0
                Zsp = 1.0
                nussp = 1.0
                for ivperp in 1:vperp.n
                    for ivpa in 1:vpa.n
                        Fs_M[ivpa,ivperp] = F_Maxwellian(denss,upars,vths,vpa.grid[ivpa],vperp.grid[ivperp])
                        F_M[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
                        C_M_exact[ivpa,ivperp] = Cssp_Maxwellian_inputs(denss,upars,vths,ms,Zs,
                                                                        dens,upar,vth,msp,Zsp,
                                                                        nussp,vpa.grid[ivpa],vperp.grid[ivperp])
                    end
                end
                fokker_planck_collision_operator_weak_form!(C_M_num,Fs_M,F_M,ms,msp,nussp,fkpl_arrays,
                                                 use_Maxwellian_Rosenbluth_coefficients=use_Maxwellian_Rosenbluth_coefficients,
                                                 algebraic_solve_for_d2Gdvperp2=algebraic_solve_for_d2Gdvperp2,
                                                 calculate_GG = false, calculate_dGdvperp=false)
                if test_numerical_conserving_terms && test_self_operator
                    # enforce the boundary conditions on CC before it is used for timestepping
                    enforce_vpavperp_BCs!(C_M_num,vpa,vperp)
                    # make ad-hoc conserving corrections for self operator using multispecies function
                    # reshaped views to the original arrays to match the multi-species interface
                    Cvpavperps = reshape(C_M_num,(vpa.n,vperp.n,species.n))
                    Fvpavperps = reshape(Fs_M,(vpa.n,vperp.n,species.n))
                    # copy correct Rosenbluth potentials into the arrays used in conserving_corrections!()
                    fkpl_arrays.rosenbluth_potentials_s[1] = fkpl_arrays.rosenbluth_potentials
                    conserving_corrections!(Cvpavperps,Fvpavperps,nussp,fkpl_arrays)
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
                dSdt = calculate_entropy_production(C_M_num,Fs_M,fkpl_arrays)

                if test_self_operator && !test_numerical_conserving_terms
                    if algebraic_solve_for_d2Gdvperp2
                        rtol, atol = 0.0, 1.0e-7
                    else
                        rtol, atol = 0.0, 1.0e-8
                    end
                    @test isapprox(dSdt, rtol ; atol=atol)
                    delta_n = get_density(C_M_num, vpa, vperp)
                    delta_upar = get_upar(C_M_num, vpa, vperp, dens)
                    delta_pressure = get_pressure(C_M_num, vpa, vperp, upar, msp)
                    delta_ppar = get_ppar(C_M_num, vpa, vperp, upar, msp)
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
                    delta_pressure = get_pressure(C_M_num, vpa, vperp, upar, msp)
                    delta_ppar = get_ppar(C_M_num, vpa, vperp, upar, msp)
                    delta_pperp = get_pperp(delta_pressure, delta_ppar)
                    rtol, atol = 0.0, 1.0e-14
                    @test isapprox(delta_n, rtol ; atol=atol)
                    rtol, atol = 0.0, 1.0e-14
                    @test isapprox(delta_upar, rtol ; atol=atol)
                    rtol, atol = 0.0, 1.0e-14
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
            slowing_down_fokker_planck_collisions_test(;
                bc = natural_boundary_condition,
                multi_species_operator_option = single_assembly_per_species)
            slowing_down_fokker_planck_collisions_test(;
                bc = zero_boundary_condition,
                multi_species_operator_option = repeat_assembly_per_species)
        end

        @testset "weak-form (multi-species) collision operator calculation" begin
            println("    - test weak-form (multi-species) collision operator calculation")
            ngrid = 17
            nelement_vpa = 4
            nelement_vperp = 2
            atol_max = 1.0e-4
            atol_L2 = 1.0e-6
            multi_species_fokker_planck_collisions_test(ngrid=ngrid,
                nelement_vpa=nelement_vpa, nelement_vperp=nelement_vperp,
                atol_max = atol_max, atol_L2=atol_L2,
                print_to_screen=print_to_screen)
        end
        @testset "weak-form (multi-species multi-reference-speed) collision operator calculation" begin
            println("    - test weak-form (multi-species multi-reference-speed) collision operator calculation")
            ngrid = 9
            nelement_vpa = 16
            nelement_vperp = 8
            atol_max = 5.0e-4
            atol_L2 = 5.0e-6
            multi_species_multi_reference_fokker_planck_collisions_test(ngrid=ngrid,
                nelement_vpa=nelement_vpa, nelement_vperp=nelement_vperp,
                atol_max = atol_max, atol_L2=atol_L2,
                print_to_screen=print_to_screen)
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
                    F_M[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
                    H_M_exact[ivpa,ivperp] = H_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
                    G_M_exact[ivpa,ivperp] = G_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
                    d2Gdvpa2_M_exact[ivpa,ivperp] = d2Gdvpa2_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
                    d2Gdvperp2_M_exact[ivpa,ivperp] = d2Gdvperp2_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
                    dGdvperp_M_exact[ivpa,ivperp] = dGdvperp_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
                    d2Gdvperpdvpa_M_exact[ivpa,ivperp] = d2Gdvperpdvpa_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
                    dHdvpa_M_exact[ivpa,ivperp] = dHdvpa_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
                    dHdvperp_M_exact[ivpa,ivperp] = dHdvperp_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
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
            multi_species_numerical_error_corrections_test(print_to_screen=print_to_screen)
        end


    end
end

end #FokkerPlanckTestsBase

using .FokkerPlanckTestsBase

FokkerPlanckTestsBase.runtests()

