export test_implicit_collisions_timing
using Plots
using LaTeXStrings
using FokkerPlanck
using HDF5
include("../test/ImplicitCollisionsTest.jl")

function save_timing_data(filename, nspecies_list,
    init_times_single, run_times_single, expected_run_times_single,
    init_times_repeat, run_times_repeat, expected_run_times_repeat)
    ntests = length(nspecies_list)
    fid = h5open(filename, "w")
    fid["ntests"] = ntests
    fid["nspecies_list"] = nspecies_list
    fid["init_times_single"] = init_times_single
    fid["run_times_single"] = run_times_single
    fid["expected_run_times_single"] = expected_run_times_single
    fid["init_times_repeat"] = init_times_repeat
    fid["run_times_repeat"] = run_times_repeat
    fid["expected_run_times_repeat"] = expected_run_times_repeat
    close(fid)
    return nothing
end

function test_implicit_collisions_timing(;nspecies_list = [1,2,3,4,5,10],
        save_HDF5=true)
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
        x = nspecies_list[i]/nspecies_list[end]
        expected_run_times_single[i] = x*0.9*run_times_single[end]
        expected_run_times_repeat[i] = (x^2)*0.9*run_times_repeat[end]
    end
    if save_HDF5
        save_timing_data("timing_test.h5", nspecies_list,
            init_times_single, run_times_single, expected_run_times_single,
            init_times_repeat, run_times_repeat, expected_run_times_repeat)
    end
    fontsize = 8
    ytick_sequence = Array([1.0e3,1.0e4,1.0e5,1.0e6])
    xlabel = L"N_{species}"
    run_times_single_label = "single assembly"
    run_times_repeat_label = "repeat assembly"
    expected_run_times_single_label = L"\propto O(N_{species})"
    expected_run_times_repeat_label = L"\propto O(N_{species}^2)"
    plot(nspecies_list, [run_times_single, run_times_repeat, expected_run_times_single, expected_run_times_repeat],
    xlabel=xlabel, label=[run_times_single_label run_times_repeat_label expected_run_times_single_label expected_run_times_repeat_label],
        ylabel="time (ms)", title = "Runtime for backward-Euler FP simulation",
        shape =:circle, xscale=:log10, yscale=:log10, xticks = (nspecies_list, nspecies_list),
        #yticks = (ytick_sequence, ytick_sequence),
        markersize = 5, linewidth=2,
        xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize,
        foreground_color_legend = nothing, background_color_legend = nothing, legend=:topleft, legend_title="Multispecies operator option",
        legend_title_font= fontsize)
    outfile = "timing_test.pdf"
    savefig(outfile)
    println(outfile)
    return nothing
end
