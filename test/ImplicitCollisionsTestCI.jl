
using Test: @test, @testset
# Tests using the backward Euler solver for
# dF/dt = C for single species, multispecies cases
# and dF/dt = C + S for the slowing down of alpha particles

# provides functions for test below to keep this script concise
include(joinpath(@__DIR__,"ImplicitCollisionsTest.jl"))

######################
# Single species tests
######################

# include expected results for single species test
include(joinpath(@__DIR__,"SingleSpeciesTestResults.jl"))

atol = 1.0e-13
@testset "Implicit collisions demo script" begin
    println("Test implicit collisions demo script")
    @testset "Gauss Legendre" begin
        println("    - test Gauss Legendre")
        for test_input_array_type in (true,false)
            output_pdf_and_grid = test_implicit_collisions(test_particle_preconditioner=true,test_numerical_conserving_terms=true,
            vth0=0.5,vperp0=1.0,vpa0=0.1, nelement_vpa=6,nelement_vperp=3,Lvpa=8.0,Lvperp=4.0, bc_vpa=natural_boundary_condition, bc_vperp=natural_boundary_condition,
                ntime=50, delta_t = 1.0, ngrid=3, test_linearised_advance=false, print_diagnostics=false, print_timing=false,
                test_external_chebyshev_grid=false, test_type=continuous_integration_test,
                test_input_array_type=test_input_array_type)
            @test isapprox(expected_gausslegendre.vpa_grid[:], output_pdf_and_grid.vpa_grid[:], atol=atol)
            @test isapprox(expected_gausslegendre.vperp_grid[:], output_pdf_and_grid.vperp_grid[:], atol=atol)
            for it in 1:2
                @test isapprox(expected_gausslegendre.pdf[:,:,:,it], output_pdf_and_grid.pdf[:,:,:,it], atol=atol)
            end
        end
    end
    @testset "Gauss Chebyshev" begin
        println("    - test Gauss Chebyshev")
        output_pdf_and_grid = test_implicit_collisions(test_particle_preconditioner=true,test_numerical_conserving_terms=true,
           vth0=0.5,vperp0=1.0,vpa0=0.1, nelement_vpa=6,nelement_vperp=3,Lvpa=8.0,Lvperp=4.0, bc_vpa=natural_boundary_condition, bc_vperp=natural_boundary_condition,
            ntime=50, delta_t = 1.0, ngrid=3, test_linearised_advance=false, print_diagnostics=false, print_timing=false,
            test_external_chebyshev_grid=true, test_type=continuous_integration_test)
        @test isapprox(expected_chebyshev.vpa_grid[:], output_pdf_and_grid.vpa_grid[:], atol=atol)
        @test isapprox(expected_chebyshev.vperp_grid[:], output_pdf_and_grid.vperp_grid[:], atol=atol)
        for it in 1:2
            @test isapprox(expected_chebyshev.pdf[:,:,:,it], output_pdf_and_grid.pdf[:,:,:,it], atol=atol)
        end
    end
end

###########################
# Multi species tests below
###########################

# include expected results for multi species test
include(joinpath(@__DIR__,"MultiSpeciesTestResults.jl"))
atol = 1.0e-13
@testset "Multi Species Implicit collisions demo script" begin
    println("Test multi species implicit collisions demo script")
    @testset "Gauss Legendre" begin
        println("    - test Gauss Legendre")
        println("        - test nspecies=1")
        for test_input_array_type in (true,false)
            # test array typing for cheapest CI test only
            output_pdf_and_grid = test_multispecies_implicit_collisions(test_particle_preconditioner=true,test_numerical_conserving_terms=true,
               vth0=[0.5],vperp0=[1.0],vpa0=[1.0],mass=[1.0],zeds=[1.0],zbeam=[0.0],density_in=[1.0],
               nelement_vpa=6,nelement_vperp=3,Lvpa=10.0,Lvperp=5.0,
               bc_vpa=natural_boundary_condition, bc_vperp=natural_boundary_condition,
               ntime=100, delta_t = 1.0, ngrid=5, test_linearised_advance=false,
               atol=1.0e-10, nonlinear_max_iterations=20, test_type=continuous_integration_test,
               print_diagnostics=false, print_timing=false, test_input_array_type=test_input_array_type)
            @test isapprox(expected_gausslegendre_nspecies_1.vpa_grid[:], output_pdf_and_grid.vpa_grid[:], atol=atol)
            @test isapprox(expected_gausslegendre_nspecies_1.vperp_grid[:], output_pdf_and_grid.vperp_grid[:], atol=atol)
            for it in 1:2
                @test isapprox(expected_gausslegendre_nspecies_1.pdf[:,:,:,it], output_pdf_and_grid.pdf[:,:,:,it], atol=atol)
            end
        end
        println("        - test nspecies=2")
        output_pdf_and_grid = test_multispecies_implicit_collisions(test_particle_preconditioner=true,test_numerical_conserving_terms=true,
            vth0=[0.5, 0.5],vperp0=[1.0, 1.0],vpa0=[1.0,0.1],mass=[1.0,2.0],zeds=[-1.0,2.0],zbeam=[0.0,0.5],density_in=[1.0,0.5],
            nelement_vpa=6,nelement_vperp=3,Lvpa=10.0,Lvperp=5.0,
            bc_vpa=natural_boundary_condition, bc_vperp=natural_boundary_condition,
            ntime=100, delta_t = 1.0, ngrid=5, test_linearised_advance=false, atol=1.0e-10, nonlinear_max_iterations=20,
            test_type=continuous_integration_test, print_diagnostics=false, print_timing=false)
        @test isapprox(expected_gausslegendre_nspecies_2.vpa_grid[:], output_pdf_and_grid.vpa_grid[:], atol=atol)
        @test isapprox(expected_gausslegendre_nspecies_2.vperp_grid[:], output_pdf_and_grid.vperp_grid[:], atol=atol)
        for it in 1:2
            @test isapprox(expected_gausslegendre_nspecies_2.pdf[:,:,:,it], output_pdf_and_grid.pdf[:,:,:,it], atol=atol)
        end
        println("        - test nspecies=3")
        output_pdf_and_grid = test_multispecies_implicit_collisions(test_particle_preconditioner=true,test_numerical_conserving_terms=true,
            vth0=[0.5, 0.5, 0.5],vperp0=[1.0, 1.0, 1.0],vpa0=[1.0,0.1,-0.5],mass=[0.5,1.0,2.0],zeds=[-1.0,1.0,2.0],zbeam=[0.0,0.5,0.1],density_in=[1.0,2.0/3.0,1.0/3.0],nelement_vpa=6,nelement_vperp=3,Lvpa=10.0,Lvperp=5.0,
            bc_vpa=natural_boundary_condition, bc_vperp=natural_boundary_condition,
            ntime=100, delta_t = 1.0, ngrid=5, test_linearised_advance=false, atol=1.0e-10, nonlinear_max_iterations=20,
            test_type=continuous_integration_test, print_diagnostics=false, print_timing=false)
        @test isapprox(expected_gausslegendre_nspecies_3.vpa_grid[:], output_pdf_and_grid.vpa_grid[:], atol=atol)
        @test isapprox(expected_gausslegendre_nspecies_3.vperp_grid[:], output_pdf_and_grid.vperp_grid[:], atol=atol)
        for it in 1:2
            @test isapprox(expected_gausslegendre_nspecies_3.pdf[:,:,:,it], output_pdf_and_grid.pdf[:,:,:,it], atol=atol)
        end
    end
end

# include expected results for slowing down CI tests
include(joinpath(@__DIR__,"SlowingDownTestResults.jl"))
atol = 1.0e-13
@testset "Slowing Down Implicit collisions demo script" begin
    println("Test slowing down implicit collisions demo script")
    @testset "Slowing down of alphas" begin
        println("    - test linearised operator")
        # test array typing for cheapest CI test only
        output_pdf_and_grid, sd_pdf, sd_errors = test_implicit_slowing_down(ntime=20,test_linearised_advance=true,
            electron_mass=(1.0/1836.0), thermal_temperature=0.01,
            nelement_vpa=32,nelement_vperp=16, ngrid=5,
            source_rate=[1.0], source_vth=[0.05], sink_rate=[100.0], sink_vth=0.05,
            print_diagnostics=false, print_timing=false, print_final_pdf=false)
        @test isapprox(expected_numerical_slowing_down.vpa_grid[:], output_pdf_and_grid.vpa_grid[:], atol=atol)
        @test isapprox(expected_numerical_slowing_down.vperp_grid[:], output_pdf_and_grid.vperp_grid[:], atol=atol)
        for it in 1:2
            @test isapprox(expected_numerical_slowing_down.pdf[:,:,:,it], output_pdf_and_grid.pdf[:,:,:,it], atol=atol)
        end
        # test relative norms of F in range [min_v,max_v]
        @test sd_errors.L2norm_relative[1] < 0.065 # L2(F-F_SD)/L2(F)
        @test sd_errors.maxnorm_relative[1] < 0.09 # max |F - F_SD| / max |F|
        @test isapprox(sd_errors.min_v[1], 0.7521206186172787, atol = atol)
        @test isapprox(sd_errors.max_v[1], 1.1, atol = atol)
        println("    - test nonlinear operator")
        # advance one timestep for a quick test
        output_pdf_and_grid, sd_pdf, sd_errors = test_implicit_slowing_down(ntime=1,test_linearised_advance=false,
            electron_mass=(1.0/1836.0), thermal_temperature=0.01,
            nelement_vpa=32,nelement_vperp=16, ngrid=5,
            source_rate=[1.0], source_vth=[0.05], sink_rate=[100.0], sink_vth=0.05,
            print_diagnostics=false, print_timing=false, print_final_pdf=false, delta_t=0.2)
        @test isapprox(expected_numerical_slowing_down_nonlinear.vpa_grid[:], output_pdf_and_grid.vpa_grid[:], atol=atol)
        @test isapprox(expected_numerical_slowing_down_nonlinear.vperp_grid[:], output_pdf_and_grid.vperp_grid[:], atol=atol)
        for it in 1:2
            @test isapprox(expected_numerical_slowing_down_nonlinear.pdf[:,:,:,it], output_pdf_and_grid.pdf[:,:,:,it], atol=atol)
        end
        # test relative norms of F in range [min_v,max_v]
        @test sd_errors.L2norm_relative[1] < 0.065 # L2(F-F_SD)/L2(F)
        @test sd_errors.maxnorm_relative[1] < 0.095 # max |F - F_SD| / max |F|
        @test isapprox(sd_errors.min_v[1], 0.7521206186172787, atol = atol)
        @test isapprox(sd_errors.max_v[1], 1.1, atol = atol)
    end
end
