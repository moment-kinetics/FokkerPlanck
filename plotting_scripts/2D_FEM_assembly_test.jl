export run_assembly_test
using Printf
using Plots
using LaTeXStrings
using Measures
using Dates
import FokkerPlanck
using FokkerPlanck.array_allocation: allocate_float
using FokkerPlanck.coordinates: finite_element_coordinate, scalar_coordinate_inputs
using FokkerPlanck.type_definitions: mk_float, mk_int
using FokkerPlanck: init_fokker_planck_collisions
using FokkerPlanck: fokker_planck_collision_operator_weak_form!
using FokkerPlanck: conserving_corrections!
using FokkerPlanck.velocity_moments: get_density, get_upar, get_pressure, get_ppar, get_pperp
using FokkerPlanck.fokker_planck_calculus: direct_integration, multipole_expansion, delta_f_multipole
using LinearAlgebra: mul!

using FokkerPlanck.fokker_planck_test: F_Maxwellian, G_Maxwellian, H_Maxwellian
using FokkerPlanck.fokker_planck_test: d2Gdvpa2_Maxwellian, d2Gdvperp2_Maxwellian, d2Gdvperpdvpa_Maxwellian, dGdvperp_Maxwellian
using FokkerPlanck.fokker_planck_test: dHdvperp_Maxwellian, dHdvpa_Maxwellian
using FokkerPlanck.fokker_planck_test: Cssp_Maxwellian_inputs
using FokkerPlanck.fokker_planck_test: print_test_data, fkpl_error_data, allocate_error_data
using FokkerPlanck.fokker_planck_test: save_fkpl_error_data

using FokkerPlanck.fokker_planck_calculus: fokkerplanck_weakform_arrays_struct
using FokkerPlanck.fokker_planck_calculus: rosenbluth_potential_boundary_data
using FokkerPlanck.fokker_planck_calculus: calculate_rosenbluth_potential_boundary_data!, calculate_rosenbluth_potential_boundary_data_exact!
using FokkerPlanck.fokker_planck_calculus: test_rosenbluth_potential_boundary_data, enforce_vpavperp_BCs!
using FokkerPlanck.fokker_planck_calculus: calculate_rosenbluth_potentials_via_elliptic_solve!

function plot_test_data(func_exact,func_num,func_err,func_name,vpa,vperp)
    @views heatmap(vperp.grid, vpa.grid, func_num[:,:], ylabel=L"v_{\|\|}", xlabel=L"v_{\perp}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string(func_name*"_num.pdf")
                savefig(outfile)
    @views heatmap(vperp.grid, vpa.grid, func_exact[:,:], ylabel=L"v_{\|\|}", xlabel=L"v_{\perp}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string(func_name*"_exact.pdf")
                savefig(outfile)
    @views heatmap(vperp.grid, vpa.grid, func_err[:,:], ylabel=L"v_{\|\|}", xlabel=L"v_{\perp}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string(func_name*"_err.pdf")
                savefig(outfile)
    return nothing
end
    
    function print_matrix(matrix,name::String,n::mk_int,m::mk_int)
        println("\n ",name," \n")
        for i in 1:n
            for j in 1:m
                @printf("%.2f ", matrix[i,j])
            end
            println("")
        end
        println("\n")
    end
    
    function print_vector(vector,name::String,m::mk_int)
        println("\n ",name," \n")
        for j in 1:m
            @printf("%.3f ", vector[j])
        end
        println("")
        println("\n")
    end 

    function test_weak_form_collisions(ngrid,nelement_vpa,nelement_vperp;
        Lvpa=12.0,Lvperp=6.0,plot_test_output=false,
        test_self_operator=true,
        standalone=false,
        use_Maxwellian_Rosenbluth_coefficients=false,
        test_numerical_conserving_terms=false,
        algebraic_solve_for_d2Gdvperp2=false,
        boundary_data_option=direct_integration)
        
        element_spacing_option = "uniform"
        println("made inputs")
        println("vpa: ngrid: ",ngrid," nelement: ",nelement_vpa, " Lvpa: ",Lvpa)
        println("vperp: ngrid: ",ngrid," nelement: ",nelement_vperp, " Lvperp: ",Lvperp)
        # create the coordinate structs
        vperp = finite_element_coordinate("vperp", scalar_coordinate_inputs(ngrid,
                                    nelement_vperp,
                                    Lvperp),
                                    element_spacing_option=element_spacing_option)
        vpa = finite_element_coordinate("vpa", scalar_coordinate_inputs(ngrid,
                                    nelement_vpa,
                                    Lvpa),
                                    element_spacing_option=element_spacing_option)
        nc_global = vpa.n*vperp.n
        start_init_time = now()
        fkpl_arrays = fokkerplanck_weakform_arrays_struct(vpa,vperp,boundary_data_option)
        KKpar2D_with_BC_terms_sparse = fkpl_arrays.KKpar2D_with_BC_terms_sparse
        KKperp2D_with_BC_terms_sparse = fkpl_arrays.KKperp2D_with_BC_terms_sparse
        lu_obj_MM = fkpl_arrays.lu_obj_MM
        finish_init_time = now()
        
        fvpavperp = Array{mk_float,2}(undef,vpa.n,vperp.n)
        fvpavperp_test = Array{mk_float,2}(undef,vpa.n,vperp.n)
        fvpavperp_err = Array{mk_float,2}(undef,vpa.n,vperp.n)
        d2fvpavperp_dvpa2_exact = Array{mk_float,2}(undef,vpa.n,vperp.n)
        d2fvpavperp_dvpa2_err = Array{mk_float,2}(undef,vpa.n,vperp.n)
        d2fvpavperp_dvpa2_num = Array{mk_float,2}(undef,vpa.n,vperp.n)
        d2fvpavperp_dvperp2_exact = Array{mk_float,2}(undef,vpa.n,vperp.n)
        d2fvpavperp_dvperp2_err = Array{mk_float,2}(undef,vpa.n,vperp.n)
        d2fvpavperp_dvperp2_num = Array{mk_float,2}(undef,vpa.n,vperp.n)
        fc = Array{mk_float,1}(undef,nc_global)
        dfc = Array{mk_float,1}(undef,nc_global)
        gc = Array{mk_float,1}(undef,nc_global)
        dgc = Array{mk_float,1}(undef,nc_global)
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                fvpavperp[ivpa,ivperp] = exp(-vpa.grid[ivpa]^2 - vperp.grid[ivperp]^2)
                d2fvpavperp_dvpa2_exact[ivpa,ivperp] = (4.0*vpa.grid[ivpa]^2 - 2.0)*exp(-vpa.grid[ivpa]^2 - vperp.grid[ivperp]^2)
                # actually d(vperp d f dvperp)/dvperp
                d2fvpavperp_dvperp2_exact[ivpa,ivperp] = 4.0*(vperp.grid[ivperp]^2 - 1.0)*exp(-vpa.grid[ivpa]^2 - vperp.grid[ivperp]^2)
            end
        end
        
        # get 1d views of fvpavperp
        fc = vec(fvpavperp)
        d2fc_dvpa2 = vec(d2fvpavperp_dvpa2_num)
        d2fc_dvperp2 = vec(d2fvpavperp_dvperp2_num)

        # multiply by KKpar2D and fill dfc
        mul!(dfc,KKpar2D_with_BC_terms_sparse,fc)
        mul!(dgc,KKperp2D_with_BC_terms_sparse,fc)
        # invert mass matrix and fill fc
        d2fc_dvpa2 .= lu_obj_MM \ dfc
        d2fc_dvperp2 .= lu_obj_MM \ dgc
        if nc_global < 30
            print_matrix(d2fvpavperp_dvpa2_num,"d2fvpavperp_dvpa2_num",vpa.n,vperp.n)
        end
        @. d2fvpavperp_dvpa2_err = abs(d2fvpavperp_dvpa2_num - d2fvpavperp_dvpa2_exact)
        println("maximum(d2fvpavperp_dvpa2_err): ",maximum(d2fvpavperp_dvpa2_err))
        @. d2fvpavperp_dvperp2_err = abs(d2fvpavperp_dvperp2_num - d2fvpavperp_dvperp2_exact)
        println("maximum(d(vperp d fvpavperp/ dvperp)/dvperp_err): ",maximum(d2fvpavperp_dvperp2_err))
        if nc_global < 30
            print_matrix(d2fvpavperp_dvpa2_err,"d2fvpavperp_dvpa2_err",vpa.n,vperp.n)
        end
        if plot_test_output
            plot_test_data(d2fvpavperp_dvpa2_exact,d2fvpavperp_dvpa2_num,d2fvpavperp_dvpa2_err,"d2fvpavperp_dvpa2",vpa,vperp)
            plot_test_data(d2fvpavperp_dvperp2_exact,d2fvpavperp_dvperp2_num,d2fvpavperp_dvperp2_err,"d2fvpavperp_dvperp2",vpa,vperp)
        end
        # test the Laplacian solve with a standard F_Maxwellian -> H_Maxwellian test
        dummy_vpavperp = Array{mk_float,2}(undef,vpa.n,vperp.n)
        Fs_M = Array{mk_float,2}(undef,vpa.n,vperp.n)
        F_M = Array{mk_float,2}(undef,vpa.n,vperp.n)
        C_M_num = allocate_float(vpa.n,vperp.n)
        C_M_exact = Array{mk_float,2}(undef,vpa.n,vperp.n)
        C_M_err = Array{mk_float,2}(undef,vpa.n,vperp.n)
        #dFdvpa_M = Array{mk_float,2}(undef,vpa.n,vperp.n)
        #dFdvperp_M = Array{mk_float,2}(undef,vpa.n,vperp.n)
        #d2Fdvperpdvpa_M = Array{mk_float,2}(undef,vpa.n,vperp.n)
        H_M_exact = Array{mk_float,2}(undef,vpa.n,vperp.n)
        H_M_num = allocate_float(vpa.n,vperp.n)
        H_M_err = Array{mk_float,2}(undef,vpa.n,vperp.n)
        G_M_exact = Array{mk_float,2}(undef,vpa.n,vperp.n)
        G_M_num = allocate_float(vpa.n,vperp.n)
        G_M_err = Array{mk_float,2}(undef,vpa.n,vperp.n)
        d2Gdvpa2_M_exact = Array{mk_float,2}(undef,vpa.n,vperp.n)
        d2Gdvpa2_M_num = allocate_float(vpa.n,vperp.n)
        d2Gdvpa2_M_err = Array{mk_float,2}(undef,vpa.n,vperp.n)
        d2Gdvperp2_M_exact = Array{mk_float,2}(undef,vpa.n,vperp.n)
        d2Gdvperp2_M_num = allocate_float(vpa.n,vperp.n)
        d2Gdvperp2_M_err = Array{mk_float,2}(undef,vpa.n,vperp.n)
        dGdvperp_M_exact = Array{mk_float,2}(undef,vpa.n,vperp.n)
        dGdvperp_M_num = allocate_float(vpa.n,vperp.n)
        dGdvperp_M_err = Array{mk_float,2}(undef,vpa.n,vperp.n)
        d2Gdvperpdvpa_M_exact = Array{mk_float,2}(undef,vpa.n,vperp.n)
        d2Gdvperpdvpa_M_num = allocate_float(vpa.n,vperp.n)
        d2Gdvperpdvpa_M_err = Array{mk_float,2}(undef,vpa.n,vperp.n)
        dHdvpa_M_exact = Array{mk_float,2}(undef,vpa.n,vperp.n)
        dHdvpa_M_num = allocate_float(vpa.n,vperp.n)
        dHdvpa_M_err = Array{mk_float,2}(undef,vpa.n,vperp.n)
        dHdvperp_M_exact = Array{mk_float,2}(undef,vpa.n,vperp.n)
        dHdvperp_M_num = allocate_float(vpa.n,vperp.n)
        dHdvperp_M_err = Array{mk_float,2}(undef,vpa.n,vperp.n)

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
                H_M_exact[ivpa,ivperp] = H_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                G_M_exact[ivpa,ivperp] = G_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                d2Gdvpa2_M_exact[ivpa,ivperp] = d2Gdvpa2_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                d2Gdvperp2_M_exact[ivpa,ivperp] = d2Gdvperp2_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                dGdvperp_M_exact[ivpa,ivperp] = dGdvperp_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                d2Gdvperpdvpa_M_exact[ivpa,ivperp] = d2Gdvperpdvpa_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                dHdvpa_M_exact[ivpa,ivperp] = dHdvpa_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                dHdvperp_M_exact[ivpa,ivperp] = dHdvperp_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
                C_M_exact[ivpa,ivperp] = Cssp_Maxwellian_inputs(denss,upars,vths,ms,
                                                                dens,upar,vth,msp,
                                                                nussp,vpa,vperp,ivpa,ivperp)
            end
        end
        rpbd_exact = rosenbluth_potential_boundary_data(vpa,vperp)

        
        # use known test function to provide exact data
        calculate_rosenbluth_potential_boundary_data_exact!(rpbd_exact,
              H_M_exact,dHdvpa_M_exact,dHdvperp_M_exact,G_M_exact,
              dGdvperp_M_exact,d2Gdvperp2_M_exact,
              d2Gdvperpdvpa_M_exact,d2Gdvpa2_M_exact,vpa,vperp)
        println("begin C calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
        
        fokker_planck_collision_operator_weak_form!(Fs_M,F_M,ms,msp,nussp,fkpl_arrays,
                                             use_Maxwellian_Rosenbluth_coefficients=use_Maxwellian_Rosenbluth_coefficients,
                                             algebraic_solve_for_d2Gdvperp2=algebraic_solve_for_d2Gdvperp2,
                                             calculate_GG = false, calculate_dGdvperp=false)
        if test_numerical_conserving_terms && test_self_operator
            # enforce the boundary conditions on CC before it is used for timestepping
            enforce_vpavperp_BCs!(fkpl_arrays.CC,vpa,vperp)
            # make ad-hoc conserving corrections
            conserving_corrections!(fkpl_arrays.CC,Fs_M,vpa,vperp,dummy_vpavperp)            
        end
        # calculate Rosenbluth potentials again as a standalone to G and dGdvperp
        calculate_rosenbluth_potentials_via_elliptic_solve!(fkpl_arrays.GG,fkpl_arrays.HH,fkpl_arrays.dHdvpa,fkpl_arrays.dHdvperp,
             fkpl_arrays.d2Gdvpa2,fkpl_arrays.dGdvperp,fkpl_arrays.d2Gdvperpdvpa,fkpl_arrays.d2Gdvperp2,F_M,
             vpa,vperp,fkpl_arrays;
             algebraic_solve_for_d2Gdvperp2=false,calculate_GG=true,calculate_dGdvperp=true)
        # extract C[Fs,Fs'] result
        # and Rosenbluth potentials for testing
        @inbounds begin
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    C_M_num[ivpa,ivperp] = fkpl_arrays.CC[ivpa,ivperp]
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
        
        init_time = Dates.value(finish_init_time - start_init_time)
        calculate_time = Dates.value(now() - finish_init_time)
        fkerr = allocate_error_data()
        println("finished C calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
        
        # test the boundary data calculation
        if !use_Maxwellian_Rosenbluth_coefficients
            max_H_err, max_dHdvpa_err, max_dHdvperp_err, max_G_err, max_dGdvperp_err,
            max_d2Gdvperp2_err, max_d2Gdvperpdvpa_err, max_d2Gdvpa2_err = test_rosenbluth_potential_boundary_data(fkpl_arrays.rpbd,rpbd_exact,vpa,vperp)
        end
        dummy_array = Array{mk_float,2}(undef,vpa.n,vperp.n)
        fkerr.H_M.max, fkerr.H_M.L2 = print_test_data(H_M_exact,H_M_num,H_M_err,"H_M",vpa,vperp,dummy_array)
        fkerr.dHdvpa_M.max, fkerr.dHdvpa_M.L2 = print_test_data(dHdvpa_M_exact,dHdvpa_M_num,dHdvpa_M_err,"dHdvpa_M",vpa,vperp,dummy_array)
        fkerr.dHdvperp_M.max, fkerr.dHdvperp_M.L2 = print_test_data(dHdvperp_M_exact,dHdvperp_M_num,dHdvperp_M_err,"dHdvperp_M",vpa,vperp,dummy_array)
        fkerr.G_M.max, fkerr.G_M.L2 = print_test_data(G_M_exact,G_M_num,G_M_err,"G_M",vpa,vperp,dummy_array)
        fkerr.d2Gdvpa2_M.max, fkerr.d2Gdvpa2_M.L2 = print_test_data(d2Gdvpa2_M_exact,d2Gdvpa2_M_num,d2Gdvpa2_M_err,"d2Gdvpa2_M",vpa,vperp,dummy_array)
        fkerr.dGdvperp_M.max, fkerr.dGdvperp_M.L2 = print_test_data(dGdvperp_M_exact,dGdvperp_M_num,dGdvperp_M_err,"dGdvperp_M",vpa,vperp,dummy_array)
        fkerr.d2Gdvperpdvpa_M.max, fkerr.d2Gdvperpdvpa_M.L2 = print_test_data(d2Gdvperpdvpa_M_exact,d2Gdvperpdvpa_M_num,d2Gdvperpdvpa_M_err,"d2Gdvperpdvpa_M",vpa,vperp,dummy_array)
        fkerr.d2Gdvperp2_M.max, fkerr.d2Gdvperp2_M.L2 = print_test_data(d2Gdvperp2_M_exact,d2Gdvperp2_M_num,d2Gdvperp2_M_err,"d2Gdvperp2_M",vpa,vperp,dummy_array)
        fkerr.C_M.max, fkerr.C_M.L2 = print_test_data(C_M_exact,C_M_num,C_M_err,"C_M",vpa,vperp,dummy_array)
        
        # calculate the entropy production
        lnfC = fkpl_arrays.rhsvpavperp
        @inbounds begin
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    lnfC[ivpa,ivperp] = Fs_M[ivpa,ivperp]*C_M_num[ivpa,ivperp]
                end
            end
        end
        dSdt = - get_density(lnfC,vpa,vperp)
        println("dSdt: $dSdt should be >0.0")
        if plot_test_output
            plot_test_data(C_M_exact,C_M_num,C_M_err,"C_M",vpa,vperp)
            plot_test_data(H_M_exact,H_M_num,H_M_err,"H_M",vpa,vperp)
            plot_test_data(dHdvpa_M_exact,dHdvpa_M_num,dHdvpa_M_err,"dHdvpa_M",vpa,vperp)
            plot_test_data(dHdvperp_M_exact,dHdvperp_M_num,dHdvperp_M_err,"dHdvperp_M",vpa,vperp)
            plot_test_data(G_M_exact,G_M_num,G_M_err,"G_M",vpa,vperp)
            plot_test_data(dGdvperp_M_exact,dGdvperp_M_num,dGdvperp_M_err,"dGdvperp_M",vpa,vperp)
            plot_test_data(d2Gdvperp2_M_exact,d2Gdvperp2_M_num,d2Gdvperp2_M_err,"d2Gdvperp2_M",vpa,vperp)
            plot_test_data(d2Gdvperpdvpa_M_exact,d2Gdvperpdvpa_M_num,d2Gdvperpdvpa_M_err,"d2Gdvperpdvpa_M",vpa,vperp)
            plot_test_data(d2Gdvpa2_M_exact,d2Gdvpa2_M_num,d2Gdvpa2_M_err,"d2Gdvpa2_M",vpa,vperp)
        end
        if test_self_operator
            delta_n = get_density(C_M_num, vpa, vperp)
            delta_upar = get_upar(C_M_num, vpa, vperp, dens)
            delta_pressure = msp*get_pressure(C_M_num, vpa, vperp, upar)
            delta_ppar = msp*get_ppar(C_M_num, vpa, vperp, upar)
            delta_pperp = get_pperp(delta_pressure, delta_ppar)
            println("delta_n: ", delta_n)
            println("delta_upar: ", delta_upar)
            println("delta_pressure: ", delta_pressure)
            fkerr.moments.delta_density = delta_n
            fkerr.moments.delta_upar = delta_upar
            fkerr.moments.delta_pressure = delta_pressure
        else
            delta_n = get_density(C_M_num, vpa, vperp)
            println("delta_n: ", delta_n)
            fkerr.moments.delta_density = delta_n
        end
        return fkerr, calculate_time, init_time
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

    function expect_timing!(expected,nelement_list,nscan,power)
        for iscan in 1:nscan
            expected[iscan] = nelement_list[iscan]^power
        end
    end
    
    function run_assembly_test(; ngrid=5, nelement_list = [8],
        plot_scan=true,
        save_HDF5 = true,
        plot_test_output = false,
        use_Maxwellian_Rosenbluth_coefficients=false,
        test_numerical_conserving_terms=false,
        algebraic_solve_for_d2Gdvperp2=false,
        test_self_operator = true,
        Lvpa = 12.0, Lvperp = 6.0,
        boundary_data_option = direct_integration,
        outpath = "")

        #ngrid = 5
        #plot_scan = true
        #plot_test_output = true#false
        #test_self_operator = true
        #nelement_list = Int[8, 16, 32, 64, 128]
        #nelement_list = Int[4, 8, 16, 32, 64]
        #nelement_list = Int[2, 4, 8]
        #nelement_list = Int[4, 8, 16, 32, 64]
        #nelement_list = Int[2, 4, 8, 16, 32]
        #nelement_list = Int[2, 4, 8, 16]
        #nelement_list = Int[100]
        #nelement_list = Int[8]
        #nelement_list = Int[4]
        nscan = size(nelement_list,1)
        max_C_err = Array{mk_float,1}(undef,nscan)
        max_H_err = Array{mk_float,1}(undef,nscan)
        max_G_err = Array{mk_float,1}(undef,nscan)
        max_dHdvpa_err = Array{mk_float,1}(undef,nscan)
        max_dHdvperp_err = Array{mk_float,1}(undef,nscan)
        max_d2Gdvperp2_err = Array{mk_float,1}(undef,nscan)
        max_d2Gdvpa2_err = Array{mk_float,1}(undef,nscan)
        max_d2Gdvperpdvpa_err = Array{mk_float,1}(undef,nscan)
        max_dGdvperp_err = Array{mk_float,1}(undef,nscan)
        L2_C_err = Array{mk_float,1}(undef,nscan)
        L2_H_err = Array{mk_float,1}(undef,nscan)
        L2_G_err = Array{mk_float,1}(undef,nscan)
        L2_dHdvpa_err = Array{mk_float,1}(undef,nscan)
        L2_dHdvperp_err = Array{mk_float,1}(undef,nscan)
        L2_d2Gdvperp2_err = Array{mk_float,1}(undef,nscan)
        L2_d2Gdvpa2_err = Array{mk_float,1}(undef,nscan)
        L2_d2Gdvperpdvpa_err = Array{mk_float,1}(undef,nscan)
        L2_dGdvperp_err = Array{mk_float,1}(undef,nscan)
        #max_d2fsdvpa2_err = Array{mk_float,1}(undef,nscan)
        #max_d2fsdvperp2_err = Array{mk_float,1}(undef,nscan)
        n_err = Array{mk_float,1}(undef,nscan)
        u_err = Array{mk_float,1}(undef,nscan)
        p_err = Array{mk_float,1}(undef,nscan)
        calculate_times = Array{mk_float,1}(undef,nscan)
        init_times = Array{mk_float,1}(undef,nscan)
        
        expected = Array{mk_float,1}(undef,nscan)
        expected_nelement_scaling!(expected,nelement_list,ngrid,nscan)
        expected_integral = Array{mk_float,1}(undef,nscan)
        expected_nelement_integral_scaling!(expected_integral,nelement_list,ngrid,nscan)
        expected_label = L"(1/N_{el})^{n_g - 1}"
        expected_integral_label = L"(1/N_{el})^{n_g +1}"
        
        expected_t_2 = Array{mk_float,1}(undef,nscan)
        expected_t_3 = Array{mk_float,1}(undef,nscan)
        expect_timing!(expected_t_2,nelement_list,nscan,2)
        expect_timing!(expected_t_3,nelement_list,nscan,3)
        expected_t_2_label = L"(N_{element})^2"
        expected_t_3_label = L"(N_{element})^3"
        
        for iscan in 1:nscan
            local nelement = nelement_list[iscan]
            nelement_vpa = 2*nelement
            nelement_vperp = nelement
            fkerr, calculate_times[iscan], init_times[iscan] = test_weak_form_collisions(ngrid,nelement_vpa,nelement_vperp,
            plot_test_output=plot_test_output,
            test_self_operator=test_self_operator,
            use_Maxwellian_Rosenbluth_coefficients=use_Maxwellian_Rosenbluth_coefficients,
            test_numerical_conserving_terms=test_numerical_conserving_terms,
            algebraic_solve_for_d2Gdvperp2=algebraic_solve_for_d2Gdvperp2,
            standalone=false, Lvpa=Lvpa, Lvperp=Lvperp, boundary_data_option=boundary_data_option)
            max_C_err[iscan], L2_C_err[iscan] = fkerr.C_M.max ,fkerr.C_M.L2
            max_H_err[iscan], L2_H_err[iscan] = fkerr.H_M.max ,fkerr.H_M.L2
            max_dHdvpa_err[iscan], L2_dHdvpa_err[iscan] = fkerr.dHdvpa_M.max ,fkerr.dHdvpa_M.L2
            max_dHdvperp_err[iscan], L2_dHdvperp_err[iscan] = fkerr.dHdvperp_M.max ,fkerr.dHdvperp_M.L2
            max_G_err[iscan], L2_G_err[iscan] = fkerr.G_M.max ,fkerr.G_M.L2
            max_dGdvperp_err[iscan], L2_dGdvperp_err[iscan] = fkerr.dGdvperp_M.max ,fkerr.dGdvperp_M.L2
            max_d2Gdvpa2_err[iscan], L2_d2Gdvpa2_err[iscan] = fkerr.d2Gdvpa2_M.max ,fkerr.d2Gdvpa2_M.L2
            max_d2Gdvperpdvpa_err[iscan], L2_d2Gdvperpdvpa_err[iscan] = fkerr.d2Gdvperpdvpa_M.max ,fkerr.d2Gdvperpdvpa_M.L2
            max_d2Gdvperp2_err[iscan], L2_d2Gdvperp2_err[iscan] = fkerr.d2Gdvperp2_M.max ,fkerr.d2Gdvperp2_M.L2
            n_err[iscan] = abs(fkerr.moments.delta_density)
            u_err[iscan] = abs(fkerr.moments.delta_upar)
            p_err[iscan] = abs(fkerr.moments.delta_pressure)
        end
        if plot_scan
            fontsize = 8
            #ytick_sequence = Array([1.0e-13,1.0e-12,1.0e-11,1.0e-10,1.0e-9,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0e-0,1.0e1])
            ytick_sequence = Array([1.0e-12,1.0e-11,1.0e-10,1.0e-9,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1])
            xlabel = L"N_{element}"
            Clabel = L"\epsilon_{\infty}(C)"
            Hlabel = L"\epsilon_{\infty}(H)"
            Glabel = L"\epsilon_{\infty}(G)"
            dHdvpalabel = L"\epsilon_{\infty}(dH/d v_{\|\|})"
            dHdvperplabel = L"\epsilon_{\infty}(dH/d v_{\perp})"
            d2Gdvperp2label = L"\epsilon_{\infty}(d^2G/d v_{\perp}^2)"
            d2Gdvpa2label = L"\epsilon_{\infty}(d^2G/d v_{\|\|}^2)"
            d2Gdvperpdvpalabel = L"\epsilon_{\infty}(d^2G/d v_{\perp} d v_{\|\|})"
            dGdvperplabel = L"\epsilon_{\infty}(dG/d v_{\perp})"
            
            #println(max_G_err,max_H_err,max_dHdvpa_err,max_dHdvperp_err,max_d2Gdvperp2_err,max_d2Gdvpa2_err,max_d2Gdvperpdvpa_err,max_dGdvperp_err, expected, expected_integral)
            plot(nelement_list, [max_C_err,max_H_err,max_G_err, expected, expected_integral],
            xlabel=xlabel, label=[Clabel Hlabel Glabel expected_label expected_integral_label], ylabel="",
             shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_list, nelement_list), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
              xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize,
              foreground_color_legend = nothing, background_color_legend = nothing, legend=:bottomleft)
            outfile = outpath*"fkpl_C_G_H_max_test_ngrid_"*string(ngrid)*"_GLL.pdf"
            savefig(outfile)
            println(outfile)
            println([max_C_err,max_H_err,max_G_err, expected, expected_integral])
            
            plot(nelement_list,  [max_dHdvpa_err, max_dHdvperp_err, max_d2Gdvperp2_err, max_d2Gdvpa2_err, max_d2Gdvperpdvpa_err, max_dGdvperp_err, expected,      expected_integral],
            xlabel=xlabel, label=[dHdvpalabel     dHdvperplabel     d2Gdvperp2label     d2Gdvpa2label     d2Gdvperpdvpalabel     dGdvperplabel     expected_label expected_integral_label], ylabel="",
             shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_list, nelement_list), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
              xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize,
              foreground_color_legend = nothing, background_color_legend = nothing, legend=:bottomleft)
            outfile = outpath*"fkpl_coeffs_max_test_ngrid_"*string(ngrid)*"_GLL.pdf"
            savefig(outfile)
            println(outfile)
            println([max_dHdvpa_err, max_dHdvperp_err, max_d2Gdvperp2_err, max_d2Gdvpa2_err, max_d2Gdvperpdvpa_err, max_dGdvperp_err, expected,      expected_integral])
            
            
            ClabelL2 = L"\epsilon_{L2}(C)"
            HlabelL2 = L"\epsilon_{L2}(H)"
            GlabelL2 = L"\epsilon_{L2}(G)"
            dHdvpalabelL2 = L"\epsilon_{L2}(dH/d v_{\|\|})"
            dHdvperplabelL2 = L"\epsilon_{L2}(dH/d v_{\perp})"
            d2Gdvperp2labelL2 = L"\epsilon_{L2}(d^2G/d v_{\perp}^2)"
            d2Gdvpa2labelL2 = L"\epsilon_{L2}(d^2G/d v_{\|\|}^2)"
            d2GdvperpdvpalabelL2 = L"\epsilon_{L2}(d^2G/d v_{\perp} d v_{\|\|})"
            dGdvperplabelL2 = L"\epsilon_{L2}(dG/d v_{\perp})"
            
            
            plot(nelement_list, [L2_C_err,L2_H_err,L2_G_err, expected, expected_integral],
            xlabel=xlabel, label=[ClabelL2 HlabelL2 GlabelL2 expected_label expected_integral_label], ylabel="",
             shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_list, nelement_list), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
              xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize,
              foreground_color_legend = nothing, background_color_legend = nothing, legend=:bottomleft)
            outfile = outpath*"fkpl_C_G_H_L2_test_ngrid_"*string(ngrid)*"_GLL.pdf"
            savefig(outfile)
            println(outfile)
            println([L2_C_err,L2_H_err,L2_G_err, expected, expected_integral])
            
            plot(nelement_list,  [L2_dHdvpa_err, L2_dHdvperp_err, L2_d2Gdvperp2_err, L2_d2Gdvpa2_err, L2_d2Gdvperpdvpa_err, L2_dGdvperp_err,  expected,      expected_integral],
            xlabel=xlabel, label=[dHdvpalabelL2  dHdvperplabelL2  d2Gdvperp2labelL2  d2Gdvpa2labelL2  d2GdvperpdvpalabelL2  dGdvperplabelL2   expected_label expected_integral_label], ylabel="",
             shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_list, nelement_list), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
              xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize,
              foreground_color_legend = nothing, background_color_legend = nothing, legend=:bottomleft)
            outfile = outpath*"fkpl_coeffs_L2_test_ngrid_"*string(ngrid)*"_GLL.pdf"
            savefig(outfile)
            println(outfile)
            println([L2_dHdvpa_err, L2_dHdvperp_err, L2_d2Gdvperp2_err, L2_d2Gdvpa2_err, L2_d2Gdvperpdvpa_err, L2_dGdvperp_err,  expected,      expected_integral])
            
            nlabel = L"|\Delta n|"
            ulabel = L"|\Delta u_{\|\|}|"
            plabel = L"|\Delta p|"
            
            if test_self_operator
                plot(nelement_list, [max_C_err, L2_C_err, n_err, u_err, p_err, expected, expected_integral],
                xlabel=xlabel, label=[Clabel ClabelL2 nlabel ulabel plabel expected_label expected_integral_label], ylabel="",
                 shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_list, nelement_list), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
                  xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize,
                  foreground_color_legend = nothing, background_color_legend = nothing, legend=:bottomleft)
                outfile = outpath*"fkpl_conservation_test_ngrid_"*string(ngrid)*"_GLL.pdf"
                savefig(outfile)
                println(outfile)
                println([max_C_err, L2_C_err, n_err, u_err, p_err, expected, expected_integral])
            else
                plot(nelement_list, [max_C_err, L2_C_err, n_err, expected, expected_integral],
                xlabel=xlabel, label=[Clabel ClabelL2 nlabel expected_label expected_integral_label], ylabel="",
                 shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_list, nelement_list), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
                  xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize,
                  foreground_color_legend = nothing, background_color_legend = nothing, legend=:bottomleft)
                outfile = outpath*"fkpl_conservation_test_ngrid_"*string(ngrid)*"_GLL.pdf"
                savefig(outfile)
                println(outfile)        
                println([max_C_err, L2_C_err, n_err, expected, expected_integral])
            end
            
            calculate_timeslabel = "time/step (ms)"
            init_timeslabel = "time/init (ms)"
            outfile = outpath*"fkpl_timing_test_ngrid_"*string(ngrid)*"_GLL.pdf"
            if boundary_data_option == direct_integration
                ytick_sequence_timing = Array([10^2,10^3,10^4,10^5,10^6])
                plot(nelement_list, [calculate_times, init_times, expected_t_2, expected_t_3],
                xlabel=xlabel, label=[calculate_timeslabel init_timeslabel expected_t_2_label expected_t_3_label], ylabel="",
                 shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_list, nelement_list), markersize = 5, linewidth=2, 
                  xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize,
                  foreground_color_legend = nothing, background_color_legend = nothing, legend=:topleft)
                println([calculate_times, init_times, expected_t_2, expected_t_3])
            else
                ytick_sequence_timing = Array([10^2,10^3,10^4,10^5])
                plot(nelement_list, [calculate_times, init_times, expected_t_2],
                xlabel=xlabel, label=[calculate_timeslabel init_timeslabel expected_t_2_label], ylabel="",
                 shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_list, nelement_list), markersize = 5, linewidth=2, 
                  xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize,
                  foreground_color_legend = nothing, background_color_legend = nothing, legend=:topleft)
                println([calculate_times, init_times, expected_t_2])
            end
            savefig(outfile)
            println(outfile)
        end
        if save_HDF5
            ncore = 1
            save_fkpl_error_data(outpath,ncore,ngrid,nelement_list,
                max_C_err, max_H_err, max_G_err, max_dHdvpa_err, max_dHdvperp_err,
                max_d2Gdvperp2_err, max_d2Gdvpa2_err, max_d2Gdvperpdvpa_err, max_dGdvperp_err, 
                L2_C_err, L2_H_err, L2_G_err, L2_dHdvpa_err, L2_dHdvperp_err, L2_d2Gdvperp2_err,
                L2_d2Gdvpa2_err, L2_d2Gdvperpdvpa_err, L2_dGdvperp_err,
                n_err, u_err, p_err, calculate_times, init_times, expected_t_2, expected_t_3,
                expected, expected_integral)
        end
    return nothing
    end

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")
    # make sure directories multipole_expansion/, etc exist before running this command
    for boundary_data_option in [multipole_expansion, delta_f_multipole, direct_integration]
        outpath = String(Symbol(boundary_data_option))*"/"
        run_assembly_test(outpath=outpath,boundary_data_option=boundary_data_option) # to ensure routines are compiled before plots are made
        run_assembly_test(ngrid=3,nelement_list=[8,16,32,64,128],plot_scan=true,
         outpath=outpath,boundary_data_option=boundary_data_option)
        run_assembly_test(ngrid=5,nelement_list=[4,8,16,32,64],plot_scan=true,
         outpath=outpath,boundary_data_option=boundary_data_option)
        run_assembly_test(ngrid=7,nelement_list=[2,4,8,16,32],plot_scan=true,
         outpath=outpath,boundary_data_option=boundary_data_option)
        run_assembly_test(ngrid=9,nelement_list=[2,4,8,16],plot_scan=true,
         outpath=outpath,boundary_data_option=boundary_data_option)
    end
end
