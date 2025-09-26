using FokkerPlanck
include("../test/ImplicitCollisionsTest.jl")

nspecies_list = [1,2,3,4,5,10]
ntests = length(nspecies_list)
init_times = zeros(ntests)
run_times = zeros(ntests)

for i in 1:ntests
     init_times[i], run_times[i] = test_implicit_collisions(;
           # initial pdf info
           vth0=1.0, vperp0=1.0, vpa0=0.1, zbeam=0.0, nspecies=nspecies_list[i],
           # grid info
           ngrid=5, nelement_vpa=8, nelement_vperp=4,
           Lvpa=10.0, Lvperp=5.0,
           # boundary condition info
           bc_vpa=natural_boundary_condition,
           bc_vperp=natural_boundary_condition,
           # time advance info
           ntime=100,delta_t=1.0,
           # nonlinea r solver options
           atol = 1.0e-10, rtol = 1.0e-10,
           nonlinear_max_iterations = 20, test_particle_preconditioner=true,
           # model options
           test_numerical_conserving_terms=true,
           boundary_data_option=delta_f_multipole,
           multi_species_operator_option=single_assembly_per_species,
           test_type=timing_test,
           test_external_chebyshev_grid=false,
           print_diagnostics=false, print_timing=true)
end
println("init_times = $init_times")
println("run_times = $run_times")