export test_rosenbluth_potentials_direct_integration

using Printf
using Plots
using LaTeXStrings
using Measures
using Dates

using FokkerPlanck.calculus: integral
using FokkerPlanck.coordinates: finite_element_coordinate, scalar_coordinate_inputs
using FokkerPlanck.fokker_planck_calculus: fokkerplanck_arrays_direct_integration_struct
using FokkerPlanck.fokker_planck_calculus: calculate_rosenbluth_potentials_via_direct_integration!
using FokkerPlanck.fokker_planck_test: d2Gdvpa2_Maxwellian, dGdvperp_Maxwellian, d2Gdvperpdvpa_Maxwellian, d2Gdvperp2_Maxwellian
using FokkerPlanck.fokker_planck_test: dHdvpa_Maxwellian, dHdvperp_Maxwellian, H_Maxwellian, G_Maxwellian
using FokkerPlanck.fokker_planck_test: F_Maxwellian, dFdvpa_Maxwellian, dFdvperp_Maxwellian
using FokkerPlanck.fokker_planck_test: d2Fdvpa2_Maxwellian, d2Fdvperpdvpa_Maxwellian, d2Fdvperp2_Maxwellian
using FokkerPlanck.fokker_planck_test: save_fkpl_integration_error_data
using FokkerPlanck.type_definitions: mk_float, mk_int
using FokkerPlanck.velocity_moments: get_pperp
using FokkerPlanck.array_allocation: allocate_float

function get_vth(pres,dens,mass)
        return sqrt(2.0*pres/(dens*mass))
end

function expected_nelement_scaling!(expected,nelement_list,ngrid,nscan)
    for iscan in 1:nscan
        expected[iscan] = (1.0/nelement_list[iscan])^(ngrid - 1)
    end
end

function expected_nelement_integral_scaling!(expected,nelement_list,ngrid,nscan)
    for iscan in 1:nscan
        expected[iscan] = (1.0/nelement_list[iscan])^(ngrid+1)
    end
end
"""
L2norm assuming the input is the 
absolution error ff_err = ff - ff_exact
We compute sqrt( int (ff_err)^2 d^3 v / int d^3 v)
where the volume of velocity space is finite
"""
function L2norm_vspace(ff_err,vpa,vperp)
    ff_ones = copy(ff_err)
    @. ff_ones = 1.0
    gg = copy(ff_err)
    @. gg = (ff_err)^2
    num = integral(@view(gg[:,:]), vpa.grid, 0, vpa.wgts, vperp.grid, 0, vperp.wgts)
    denom = integral(@view(ff_ones[:,:]), vpa.grid, 0, vpa.wgts, vperp.grid, 0, vperp.wgts)
    L2norm = sqrt(num/denom)
    return L2norm
end


function init_grids(nelement,ngrid)
    element_spacing_option = "uniform"
    # define inputs needed for the test
    Lvpa = 12.0 #physical box size in reference units 
    Lvperp = 6.0 #physical box size in reference units 
    
    element_spacing_option = "uniform"
    # create the coordinate structs
    vperp = finite_element_coordinate("vperp", scalar_coordinate_inputs(ngrid,
                                nelement,
                                Lvperp),
                                element_spacing_option=element_spacing_option)
    vpa = finite_element_coordinate("vpa", scalar_coordinate_inputs(ngrid,
                                nelement,
                                Lvpa),
                                element_spacing_option=element_spacing_option)
    return vpa, vperp
end

test_Lagrange_integral = false #true
test_Lagrange_integral_scan = true

function test_Lagrange_Rosenbluth_potentials(ngrid,nelement; standalone=true)
    # set up grids for input Maxwellian
    vpa, vperp =  init_grids(nelement,ngrid)
    # set up necessary inputs for collision operator functions 
    nvperp = vperp.n
    nvpa = vpa.n
    println("beginning allocation   ", Dates.format(now(), dateformat"H:MM:SS"))
    
    fs_in = Array{mk_float,2}(undef,nvpa,nvperp)
    
    dfsdvpa_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
    d2fsdvpa2_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
    dfsdvperp_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
    d2fsdvperpdvpa_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
    d2fsdvperp2_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
    dfsdvpa_err = Array{mk_float,2}(undef,nvpa,nvperp)
    d2fsdvpa2_err = Array{mk_float,2}(undef,nvpa,nvperp)
    dfsdvperp_err = Array{mk_float,2}(undef,nvpa,nvperp)
    d2fsdvperpdvpa_err = Array{mk_float,2}(undef,nvpa,nvperp)
    d2fsdvperp2_err = Array{mk_float,2}(undef,nvpa,nvperp)
    
    GG_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
    GG_err = allocate_float(nvpa,nvperp)
    d2Gdvpa2_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
    d2Gdvpa2_err = allocate_float(nvpa,nvperp)
    dGdvperp_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
    dGdvperp_err = allocate_float(nvpa,nvperp)
    d2Gdvperpdvpa_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
    d2Gdvperpdvpa_err = allocate_float(nvpa,nvperp)
    d2Gdvperp2_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
    d2Gdvperp2_err = allocate_float(nvpa,nvperp)
    
    HH_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
    HH_err = allocate_float(nvpa,nvperp)
    dHdvpa_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
    dHdvpa_err = allocate_float(nvpa,nvperp)
    dHdvperp_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
    dHdvperp_err = allocate_float(nvpa,nvperp)
    
    println("setting up input arrays   ", Dates.format(now(), dateformat"H:MM:SS"))
    
    # set up test Maxwellian
    denss = 1.0 #3.0/4.0
    upars = 0.0 #2.0/3.0
    press = 1.0 #2.0/3.0
    ppars = 1.0 #2.0/3.0
    pperps = get_pperp(press, ppars)
    ms = 1.0
    vths = get_vth(press,denss,ms)
    
    for ivperp in 1:nvperp
        for ivpa in 1:nvpa
            fs_in[ivpa,ivperp] = F_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp) #(denss/vths^3)*exp( - ((vpa.grid[ivpa]-upar)^2 + vperp.grid[ivperp]^2)/vths^2 ) 
            dfsdvpa_Maxwell[ivpa,ivperp] = dFdvpa_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
            dfsdvperp_Maxwell[ivpa,ivperp] = dFdvperp_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
            d2fsdvperpdvpa_Maxwell[ivpa,ivperp] = d2Fdvperpdvpa_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
            
            GG_Maxwell[ivpa,ivperp] = G_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
            d2Gdvpa2_Maxwell[ivpa,ivperp] = d2Gdvpa2_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
            dGdvperp_Maxwell[ivpa,ivperp] = dGdvperp_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
            d2Gdvperpdvpa_Maxwell[ivpa,ivperp] = d2Gdvperpdvpa_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
            d2Gdvperp2_Maxwell[ivpa,ivperp] = d2Gdvperp2_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
            HH_Maxwell[ivpa,ivperp] = H_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
            dHdvperp_Maxwell[ivpa,ivperp] = dHdvperp_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
            dHdvpa_Maxwell[ivpa,ivperp] = dHdvpa_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
        end
    end
    
    # initialise the weights
    fokkerplanck_arrays = fokkerplanck_arrays_direct_integration_struct(vperp,vpa)
    fka = fokkerplanck_arrays

    
    # calculate the potentials by direct integration
    calculate_rosenbluth_potentials_via_direct_integration!(fka.GG,fka.HH,fka.dHdvpa,fka.dHdvperp,
             fka.d2Gdvpa2,fka.dGdvperp,fka.d2Gdvperpdvpa,fka.d2Gdvperp2,fs_in,
             vpa,vperp,fka)
            
    # error analysis of distribution function
    println("finished integration   ", Dates.format(now(), dateformat"H:MM:SS"))
    @. dfsdvpa_err = abs(fka.dfdvpa - dfsdvpa_Maxwell)
    max_dfsdvpa_err = maximum(dfsdvpa_err)
    println("max_dfsdvpa_err: ",max_dfsdvpa_err)
    @. dfsdvperp_err = abs(fka.dfdvperp - dfsdvperp_Maxwell)
    max_dfsdvperp_err = maximum(dfsdvperp_err)
    println("max_dfsdvperp_err: ",max_dfsdvperp_err)
    @. d2fsdvperpdvpa_err = abs(fka.d2fdvperpdvpa - d2fsdvperpdvpa_Maxwell)
    max_d2fsdvperpdvpa_err = maximum(d2fsdvperpdvpa_err)
    println("max_d2fsdvperpdvpa_err: ",max_d2fsdvperpdvpa_err)
    
    plot_dHdvpa = false #true
    plot_dHdvperp = false #true
    plot_d2Gdvperp2 = false #true
    plot_d2Gdvperpdvpa = false #true
    plot_dGdvperp = false #true
    plot_d2Gdvpa2 = false #true
    
    @. GG_err = abs(fka.GG - GG_Maxwell)
    max_GG_err, max_GG_index = findmax(GG_err)
    println("max_GG_err: ",max_GG_err," ",max_GG_index)
    println("spot check GG_err: ",GG_err[end,end], " GG: ",fka.GG[end,end])
    
    @. HH_err = abs(fka.HH - HH_Maxwell)
    max_HH_err, max_HH_index = findmax(HH_err)
    println("max_HH_err: ",max_HH_err," ",max_HH_index)
    println("spot check HH_err: ",HH_err[end,end], " HH: ",fka.HH[end,end])
    @. dHdvperp_err = abs(fka.dHdvperp - dHdvperp_Maxwell)
    max_dHdvperp_err, max_dHdvperp_index = findmax(dHdvperp_err)
    println("max_dHdvperp_err: ",max_dHdvperp_err," ",max_dHdvperp_index)
    println("spot check dHdvperp_err: ",dHdvperp_err[end,end], " dHdvperp: ",fka.dHdvperp[end,end])
    @. dHdvpa_err = abs(fka.dHdvpa - dHdvpa_Maxwell)
    max_dHdvpa_err, max_dHdvpa_index = findmax(dHdvpa_err)
    println("max_dHdvpa_err: ",max_dHdvpa_err," ",max_dHdvpa_index)
    println("spot check dHdvpa_err: ",dHdvpa_err[end,end], " dHdvpa: ",fka.dHdvpa[end,end])
    
    if plot_dHdvpa
        @views heatmap(vperp.grid, vpa.grid, dHspdvpa[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string("fkpl_dHdvpa_lagrange.pdf")
                savefig(outfile)
        @views heatmap(vperp.grid, vpa.grid, dHdvpa_Maxwell[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string("fkpl_dHdvpa_Maxwell.pdf")
                savefig(outfile)
            @views heatmap(vperp.grid, vpa.grid, dHdvpa_err[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string("fkpl_dHdvpa_err.pdf")
                savefig(outfile)
    end
    if plot_dHdvperp
        @views heatmap(vperp.grid, vpa.grid, dHspdvperp[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string("fkpl_dHdvperp_lagrange.pdf")
                savefig(outfile)
        @views heatmap(vperp.grid, vpa.grid, dHdvperp_Maxwell[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string("fkpl_dHdvperp_Maxwell.pdf")
                savefig(outfile)
            @views heatmap(vperp.grid, vpa.grid, dHdvperp_err[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string("fkpl_dHdvperp_err.pdf")
                savefig(outfile)
    end
    @. d2Gdvperp2_err = abs(fka.d2Gdvperp2 - d2Gdvperp2_Maxwell)
    max_d2Gdvperp2_err, max_d2Gdvperp2_index = findmax(d2Gdvperp2_err)
    println("max_d2Gdvperp2_err: ",max_d2Gdvperp2_err," ",max_d2Gdvperp2_index)
    println("spot check d2Gdvperp2_err: ",d2Gdvperp2_err[end,end], " d2Gdvperp2: ",fka.d2Gdvperp2[end,end])
    if plot_d2Gdvperp2
        @views heatmap(vperp.grid, vpa.grid, d2Gspdvperp2[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string("fkpl_d2Gdvperp2_lagrange.pdf")
                savefig(outfile)
        @views heatmap(vperp.grid, vpa.grid, d2Gdvperp2_Maxwell[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string("fkpl_d2Gdvperp2_Maxwell.pdf")
                savefig(outfile)
            @views heatmap(vperp.grid, vpa.grid, d2Gdvperp2_err[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string("fkpl_d2Gdvperp2_err.pdf")
                savefig(outfile)
    end
    @. d2Gdvperpdvpa_err = abs(fka.d2Gdvperpdvpa - d2Gdvperpdvpa_Maxwell)
    max_d2Gdvperpdvpa_err, max_d2Gdvperpdvpa_index = findmax(d2Gdvperpdvpa_err)
    println("max_d2Gdvperpdvpa_err: ",max_d2Gdvperpdvpa_err," ",max_d2Gdvperpdvpa_index)
    println("spot check d2Gdvperpdpva_err: ",d2Gdvperpdvpa_err[end,end], " d2Gdvperpdvpa: ",fka.d2Gdvperpdvpa[end,end])
    if plot_d2Gdvperpdvpa
        @views heatmap(vperp.grid, vpa.grid, d2Gspdvperpdvpa[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string("fkpl_d2Gdvperpdvpa_lagrange.pdf")
                savefig(outfile)
        @views heatmap(vperp.grid, vpa.grid, d2Gdvperpdvpa_Maxwell[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string("fkpl_d2Gdvperpdvpa_Maxwell.pdf")
                savefig(outfile)
            @views heatmap(vperp.grid, vpa.grid, d2Gdvperpdvpa_err[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string("fkpl_d2Gdvperpdvpa_err.pdf")
                savefig(outfile)
    end
    @. dGdvperp_err = abs(fka.dGdvperp - dGdvperp_Maxwell)
    max_dGdvperp_err, max_dGdvperp_index = findmax(dGdvperp_err)
    println("max_dGdvperp_err: ",max_dGdvperp_err," ",max_dGdvperp_index)
    println("spot check dGdvperp_err: ",dGdvperp_err[end,end], " dGdvperp: ",fka.dGdvperp[end,end])
    if plot_dGdvperp
        @views heatmap(vperp.grid, vpa.grid, dGspdvperp[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string("fkpl_dGdvperp_lagrange.pdf")
                savefig(outfile)
        @views heatmap(vperp.grid, vpa.grid, dGdvperp_Maxwell[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string("fkpl_dGdvperp_Maxwell.pdf")
                savefig(outfile)
            @views heatmap(vperp.grid, vpa.grid, dGdvperp_err[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string("fkpl_dGdvperp_err.pdf")
                savefig(outfile)
    end
    @. d2Gdvpa2_err = abs(fka.d2Gdvpa2 - d2Gdvpa2_Maxwell)
    max_d2Gdvpa2_err, max_d2Gdvpa2_index = findmax(d2Gdvpa2_err)
    println("max_d2Gdvpa2_err: ",max_d2Gdvpa2_err," ",max_d2Gdvpa2_index)
    println("spot check d2Gdvpa2_err: ",d2Gdvpa2_err[end,end], " d2Gdvpa2: ",fka.d2Gdvpa2[end,end])
    if plot_d2Gdvpa2
        @views heatmap(vperp.grid, vpa.grid, d2Gspdvpa2[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string("fkpl_d2Gdvpa2_lagrange.pdf")
                savefig(outfile)
        @views heatmap(vperp.grid, vpa.grid, d2Gdvpa2_Maxwell[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string("fkpl_d2Gdvpa2_Maxwell.pdf")
                savefig(outfile)
            @views heatmap(vperp.grid, vpa.grid, d2Gdvpa2_err[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string("fkpl_d2Gdvpa2_err.pdf")
                savefig(outfile)
    end
    #println(maximum(G_err), maximum(H_err), maximum(dHdvpa_err), maximum(dHdvperp_err), maximum(d2Gdvperp2_err), maximum(d2Gdvpa2_err), maximum(d2Gdvperpdvpa_err), maximum(dGdvperp_err))
    results = (maximum(GG_err), maximum(HH_err), maximum(dHdvpa_err), maximum(dHdvperp_err), maximum(d2Gdvperp2_err), maximum(d2Gdvpa2_err), maximum(d2Gdvperpdvpa_err), maximum(dGdvperp_err),
    maximum(dfsdvpa_err), maximum(dfsdvperp_err), maximum(d2fsdvperpdvpa_err))
    return results 
end

function test_rosenbluth_potentials_direct_integration(;ngrid=5,nelement_list=[2],plot_scan=true,save_HDF5=true)
    if size(nelement_list,1) == 1
        nelement = nelement_list[1]
        test_Lagrange_Rosenbluth_potentials(ngrid,nelement,standalone=true)
    else
        nscan = size(nelement_list,1)
        max_G_err = Array{mk_float,1}(undef,nscan)
        max_H_err = Array{mk_float,1}(undef,nscan)
        max_dHdvpa_err = Array{mk_float,1}(undef,nscan)
        max_dHdvperp_err = Array{mk_float,1}(undef,nscan)
        max_d2Gdvperp2_err = Array{mk_float,1}(undef,nscan)
        max_d2Gdvpa2_err = Array{mk_float,1}(undef,nscan)
        max_d2Gdvperpdvpa_err = Array{mk_float,1}(undef,nscan)
        max_dGdvperp_err = Array{mk_float,1}(undef,nscan)
        max_dfsdvpa_err = Array{mk_float,1}(undef,nscan)
        max_dfsdvperp_err = Array{mk_float,1}(undef,nscan)
        max_d2fsdvperpdvpa_err = Array{mk_float,1}(undef,nscan)
        
        expected = Array{mk_float,1}(undef,nscan)
        expected_nelement_scaling!(expected,nelement_list,ngrid,nscan)
        expected_integral = Array{mk_float,1}(undef,nscan)
        expected_nelement_integral_scaling!(expected_integral,nelement_list,ngrid,nscan)
        
        expected_label = L"(1/N_{el})^{n_g - 1}"
        expected_integral_label = L"(1/N_{el})^{n_g +1}"
        
        for iscan in 1:nscan
            local nelement = nelement_list[iscan]
            ((max_G_err[iscan], max_H_err[iscan], max_dHdvpa_err[iscan],
            max_dHdvperp_err[iscan], max_d2Gdvperp2_err[iscan],
            max_d2Gdvpa2_err[iscan], max_d2Gdvperpdvpa_err[iscan],
            max_dGdvperp_err[iscan], max_dfsdvpa_err[iscan],
            max_dfsdvperp_err[iscan], max_d2fsdvperpdvpa_err[iscan])
            = test_Lagrange_Rosenbluth_potentials(ngrid,nelement,standalone=false))
        end
        if plot_scan
            fontsize = 8
            ytick_sequence = Array([1.0e-13,1.0e-12,1.0e-11,1.0e-10,1.0e-9,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0e-0,1.0e1])
            xlabel = L"N_{element}"
            dHdvpalabel = L"\epsilon(dH/d v_{\|\|})"
            dHdvperplabel = L"\epsilon(dH/d v_{\perp})"
            d2Gdvperp2label = L"\epsilon(d^2G/d v_{\perp}^2)"
            d2Gdvpa2label = L"\epsilon(d^2G/d v_{\|\|}^2)"
            d2Gdvperpdvpalabel = L"\epsilon(d^2G/d v_{\perp} d v_{\|\|})"
            dGdvperplabel = L"\epsilon(dG/d v_{\perp})"
            #println(max_G_err,max_H_err,max_dHdvpa_err,max_dHdvperp_err,max_d2Gdvperp2_err,max_d2Gdvpa2_err,max_d2Gdvperpdvpa_err,max_dGdvperp_err, expected)
            plot(nelement_list, [max_dHdvpa_err,max_dHdvperp_err,max_d2Gdvperp2_err,max_d2Gdvpa2_err,max_d2Gdvperpdvpa_err,max_dGdvperp_err, expected, expected_integral],
            xlabel=xlabel, label=[dHdvpalabel dHdvperplabel d2Gdvperp2label d2Gdvpa2label d2Gdvperpdvpalabel dGdvperplabel expected_label expected_integral_label], ylabel="",
             shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_list, nelement_list), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
              xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize,
              foreground_color_legend = nothing, background_color_legend = nothing, legend=:bottomleft)
            #outfile = "fkpl_coeffs_numerical_lagrange_integration_test_ngrid_"*string(ngrid)*".pdf"
            outfile = "fkpl_essential_coeffs_numerical_lagrange_integration_test_ngrid_"*string(ngrid)*"_GLL.pdf"
            savefig(outfile)
            println(outfile)
            
            HHlabel = L"\epsilon(H)"
            GGlabel = L"\epsilon(G)"
            #println(max_G_err,max_H_err,max_dHdvpa_err,max_dHdvperp_err,max_d2Gdvperp2_err,max_d2Gdvpa2_err,max_d2Gdvperpdvpa_err,max_dGdvperp_err, expected)
            plot(nelement_list, [max_H_err, max_G_err, expected, expected_integral],
            xlabel=xlabel, label=[HHlabel GGlabel expected_label expected_integral_label], ylabel="",
             shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_list, nelement_list), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
              xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize,
              foreground_color_legend = nothing, background_color_legend = nothing, legend=:bottomleft)
            #outfile = "fkpl_coeffs_numerical_lagrange_integration_test_ngrid_"*string(ngrid)*".pdf"
            outfile = "fkpl_potentials_numerical_lagrange_integration_test_ngrid_"*string(ngrid)*"_GLL.pdf"
            savefig(outfile)
            println(outfile)
            
            dfsdvpa_label = L"\epsilon(d F_s / d v_{\|\|})"
            dfsdvperp_label = L"\epsilon(d F_s /d v_{\perp})"
            d2fsdvperpdvpa_label = L"\epsilon(d^2 F_s /d v_{\perp}d v_{\|\|})"
            plot(nelement_list, [max_dfsdvpa_err,max_dfsdvperp_err,max_d2fsdvperpdvpa_err,expected],
            xlabel=xlabel, label=[dfsdvpa_label dfsdvperp_label d2fsdvperpdvpa_label expected_label], ylabel="",
             shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_list, nelement_list), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
              xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize,
              foreground_color_legend = nothing, background_color_legend = nothing, legend=:bottomleft)
            #outfile = "fkpl_coeffs_numerical_lagrange_integration_test_ngrid_"*string(ngrid)*".pdf"
            outfile = "fkpl_fs_numerical_test_ngrid_"*string(ngrid)*"_GLL.pdf"
            savefig(outfile)
            println(outfile)
        end
        if save_HDF5
            outdir = ""
            ncore = 1
            save_fkpl_integration_error_data(outdir, ncore, ngrid, nelement_list,
                max_dfsdvpa_err, max_dfsdvperp_err, max_d2fsdvperpdvpa_err,
                max_H_err, max_G_err, max_dHdvpa_err, max_dHdvperp_err,
                max_d2Gdvperp2_err, max_d2Gdvpa2_err, max_d2Gdvperpdvpa_err, max_dGdvperp_err, 
                expected, expected_integral)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")
    ngrid = 5
    nelement_list = [2,4,8,16,32]
    plot_scan = true
    test_rosenbluth_potentials_direct_integration(ngrid=ngrid,nelement_list=nelement_list,plot_scan=plot_scan)
end 
