module VelocityIntegralTests

using Test: @testset, @test
using FokkerPlanck.coordinates: finite_element_coordinate, scalar_coordinate_inputs
using FokkerPlanck.velocity_moments: get_density, get_upar, get_ppar, get_pressure
using FokkerPlanck.array_allocation: allocate_float

function runtests()
    @testset "velocity integral tests" begin
        println("velocity integral tests")

        # Tolerance for tests
        atol = 1.0e-13

        # define inputs needed for the test
        ngrid = 17 #number of points per element 
        nelement = 20 # number of elements per rank
        Lvpa = 18.0 #physical box size in reference units 
        Lvperp = 9.0 #physical box size in reference units 
        element_spacing_option = "uniform"
        # create the coordinate structs
        vr = finite_element_coordinate("vperp1d", scalar_coordinate_inputs(1,
                                1,
                                1.0),
                                element_spacing_option=element_spacing_option)
        vz = finite_element_coordinate("vpa1d", scalar_coordinate_inputs(ngrid,
                                nelement,
                                Lvpa),
                                element_spacing_option=element_spacing_option)
        vperp = finite_element_coordinate("vperp", scalar_coordinate_inputs(ngrid,
                                nelement,
                                Lvperp),
                                element_spacing_option=element_spacing_option)
        vpa = finite_element_coordinate("vpa", scalar_coordinate_inputs(ngrid,
                                nelement,
                                Lvpa),
                                element_spacing_option=element_spacing_option)
        
        dfn = allocate_float(vpa.n,vperp.n)
        dfn1D = allocate_float(vz.n, vr.n)

        function pressure(ppar,pperp)
            pres = (1.0/3.0)*(ppar + 2.0*pperp) 
            return pres
        end

        @testset "2D isotropic Maxwellian" begin
            # assign a known isotropic Maxwellian distribution in normalised units
            dens = 3.0/4.0
            upar = 2.0/3.0
            ppar = 2.0/3.0
            pperp = 2.0/3.0
            pres = pressure(ppar,pperp)
            mass = 1.0
            vth = sqrt(2.0*pres/(dens*mass))
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    vpa_val = vpa.grid[ivpa]
                    vperp_val = vperp.grid[ivperp]
                    dfn[ivpa,ivperp] = (dens/vth^3/π^1.5)*exp( - ((vpa_val-upar)^2 + vperp_val^2)/vth^2 )
                end
            end

            # now check that we can extract the correct moments from the distribution

            dens_test = get_density(dfn,vpa,vperp)
            upar_test = get_upar(dfn,vpa,vperp,dens_test)
            pres_test = get_pressure(dfn,vpa,vperp,upar_test)
            @test isapprox(dens_test, dens; atol=atol)
            @test isapprox(upar_test, upar; atol=atol)
            @test isapprox(pres_test, pres; atol=atol)
        end

        @testset "1D Maxwellian" begin
            dens = 3.0/4.0
            upar = 2.0/3.0
            ppar = 2.0/3.0 
            mass = 1.0
            vth = sqrt(2.0*ppar/(dens*mass))
            for ivz in 1:vz.n
                for ivr in 1:vr.n
                    vz_val = vz.grid[ivz]
                    dfn1D[ivz,ivr] = (dens/vth/sqrt(π))*exp( - ((vz_val-upar)^2)/vth^2 )
                end
            end
            dens_test = get_density(dfn1D,vz,vr)
            upar_test = get_upar(dfn1D,vz,vr,dens_test)
            ppar_test = get_ppar(dfn1D,vz,vr,upar_test)
            @test isapprox(dens_test, dens; atol=atol)
            @test isapprox(upar_test, upar; atol=atol)
            @test isapprox(ppar_test, ppar; atol=atol)
        end

        @testset "biMaxwellian" begin
            # assign a known biMaxwellian distribution in normalised units
            dens = 3.0/4.0
            upar = 2.0/3.0
            ppar = 4.0/5.0
            pperp = 1.0/4.0 
            mass = 1.0
            vthpar = sqrt(2.0*ppar/(dens*mass))
            vthperp = sqrt(2.0*pperp/(dens*mass))
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    vpa_val = vpa.grid[ivpa]
                    vperp_val = vperp.grid[ivperp]
                    dfn[ivpa,ivperp] = (dens/(vthpar*vthperp^2)/π^1.5)*exp( - ((vpa_val-upar)^2)/vthpar^2 - (vperp_val^2)/vthperp^2 )
                end
            end

            # now check that we can extract the correct moments from the distribution

            dens_test = get_density(dfn,vpa,vperp)
            upar_test = get_upar(dfn,vpa,vperp,dens_test)
            ppar_test = get_ppar(dfn,vpa,vperp,upar_test)

            @test isapprox(dens_test, dens; atol=atol)
            @test isapprox(upar_test, upar; atol=atol)
            @test isapprox(ppar_test, ppar; atol=atol)
        end
    end
end 

end # VelocityIntegralTests

using .VelocityIntegralTests

VelocityIntegralTests.runtests()
