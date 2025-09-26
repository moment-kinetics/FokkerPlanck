
using Plots
using LaTeXStrings
using FokkerPlanck
include("../test/ImplicitCollisionsTest.jl")

nspecies_list = [1,2,3,4,5,10]
#nspecies_list = [1,2]
ntests = length(nspecies_list)
init_times_single = zeros(ntests)
run_times_single = zeros(ntests)
expected_run_times_single = zeros(ntests)
init_times_repeat = zeros(ntests)
run_times_repeat = zeros(ntests)
expected_run_times_repeat = zeros(ntests)

for i in 1:ntests
     init_times_single[i], run_times_single[i] = test_implicit_collisions(;
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
     init_times_repeat[i], run_times_repeat[i] = test_implicit_collisions(;
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
           multi_species_operator_option=repeat_assembly_per_species,
           test_type=timing_test,
           test_external_chebyshev_grid=false,
           print_diagnostics=false, print_timing=true)
end
println("init_times_single = $init_times_single")
println("run_times_single = $run_times_single")
println("init_times_repeat = $init_times_repeat")
println("run_times_repeat = $run_times_repeat")
for i in 1:ntests
    expected_run_times_single[i] = (i/ntests)*run_times_single[end]
    expected_run_times_repeat[i] = ((i/ntests)^2)*run_times_repeat[end]
end
fontsize = 8
ytick_sequence = Array([1.0e3,1.0e4,1.0e5,1.0e6])
xlabel = L"N_{species}"
run_times_single_label = "single"
run_times_repeat_label = "repeat"
expected_run_times_single_label = L"\propto O(N_{species})"
expected_run_times_repeat_label = L"\propto O(N_{species}^2)"
plot(nspecies_list, [run_times_single, run_times_repeat, expected_run_times_single, expected_run_times_repeat],
xlabel=xlabel, label=[run_times_single_label run_times_repeat_label expected_run_times_single_label expected_run_times_repeat_label], ylabel="",
    shape =:circle, xscale=:log10, yscale=:log10, xticks = (nspecies_list, nspecies_list),
    #yticks = (ytick_sequence, ytick_sequence),
    markersize = 5, linewidth=2,
    xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize,
    foreground_color_legend = nothing, background_color_legend = nothing, legend=:bottomleft)
outfile = "timing_test.pdf"
savefig(outfile)
println(outfile)
