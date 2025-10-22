using Plots
using LaTeXStrings
using FokkerPlanck
using HDF5
include("../test/ImplicitCollisionsTest.jl")

function save_pdf_data(filename, pdf, vpa, vperp)
    fid = h5open(filename, "w")
    fid["vpa"] = vpa
    fid["vperp"] = vperp
    fid["pdf"] = pdf
    close(fid)
    return nothing
end

function plot_slowing_down_results(; ntime=20, nelement_vpa=32,nelement_vperp=16, ngrid=7, source_rate=[1.0], source_vth=[0.05], sink_rate=[100.0], sink_vth=0.05, electron_mass=1.0/1836.0, thermal_temperature=0.01)
    pdfdata, sd_pdf = test_implicit_slowing_down(ntime=ntime,test_linearised_advance=true,
                    nelement_vpa=nelement_vpa,nelement_vperp=nelement_vperp,ngrid=ngrid,
                    source_rate=source_rate,source_vth=source_vth,sink_rate=sink_rate,sink_vth=sink_vth,
                    electron_mass = electron_mass, thermal_temperature = thermal_temperature)
    vpa_grid = pdfdata.vpa_grid
    vperp_grid = pdfdata.vperp_grid
    save_pdf_data("slowing_down_pdf", sd_pdf, vpa_grid, vperp_grid)
    save_pdf_data("linear_C_pdf", pdfdata.pdf[:,:,1,2], vpa_grid, vperp_grid)
    pdfdata_nl, sd_pdf = test_implicit_slowing_down(ntime=ntime, test_linearised_advance=false,
                    nelement_vpa=nelement_vpa,nelement_vperp=nelement_vperp,ngrid=ngrid,
                    source_rate=source_rate,source_vth=source_vth,sink_rate=sink_rate,sink_vth=sink_vth,
                    electron_mass = electron_mass, thermal_temperature = thermal_temperature)
    save_pdf_data("nonlinear_C_pdf", pdfdata_nl.pdf[:,:,1,2], vpa_grid, vperp_grid)

    ivpamid = Int(floor(length(vpa_grid)/2) + 1)
    fontsize = 8
    xlabel = L"v_{\perp}"
    sdlabel = "SD"
    linlabel = "L"
    nllabel = "NL"
    outfiles = [["pdf_lin_plot.pdf", "pdf_log_plot.pdf"],["pdf_lin_plot_limit.pdf", "pdf_log_plot_limit.pdf"]]
    for (i, (xlims, ylims)) in enumerate(((:auto,:auto),((0.5,Inf),(1.0e-15,0.003))))
        for (j, yscale) in enumerate((:identity, :log10))
            plot(vperp_grid, [abs.(sd_pdf[ivpamid,:].+1.0e-15), abs.(pdfdata.pdf[ivpamid,:,1,2].+1.0e-15), abs.(pdfdata_nl.pdf[ivpamid,:,1,2].+1.0e-15)],
                xlabel=xlabel, label=[sdlabel linlabel nllabel],
                ylabel="", title = L"F(v_{\|\|} = 0, v_\perp)",
                #shape =:circle,
                xlims=xlims,
                ylims=ylims,
                xscale=:identity, yscale=yscale,
                #xticks = (nspecies_list, nspecies_list),
                #yticks = (ytick_sequence, ytick_sequence),
                markersize = 5, linewidth=2,
                xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize,
                foreground_color_legend = nothing, background_color_legend = nothing, legend=:topright, legend_title="",
                legend_title_font= fontsize)
            outfile = outfiles[i][j]
            savefig(outfile)
            println(outfile)
        end
    end
    return nothing
end
