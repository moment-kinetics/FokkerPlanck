using FokkerPlanck.coordinates: finite_element_coordinate, scalar_coordinate_inputs
using Test: @test, @testset

function test_coordinates(coord,coordnew)
    @test coord.name == coordnew.name
    @test coord.n == coordnew.n
    @test coord.ngrid == coordnew.ngrid
    @test coord.nelement == coordnew.nelement
    @test coord.bc == coordnew.bc
    @test isapprox(coord.L,coordnew.L,atol=2.0e-15)
    @test isapprox(coord.grid,coordnew.grid,atol=2.0e-15)
    @test isapprox(coord.wgts,coordnew.wgts,atol=2.0e-14) # relax tolerance here as wgts computed by different method
    @test isapprox(coord.element_scale,coordnew.element_scale,atol=2.0e-15)
    @test isapprox(coord.element_shift,coordnew.element_shift,atol=2.0e-15)
    @test isapprox(coord.element_boundaries,coordnew.element_boundaries,atol=2.0e-15)
    @test isapprox(coord.igrid,coordnew.igrid,atol=0)
    @test isapprox(coord.igrid_full,coordnew.igrid_full,atol=0)
    @test isapprox(coord.imin,coordnew.imin,atol=0)
    @test isapprox(coord.imax,coordnew.imax,atol=0)
    for j in 1:coord.nelement
        test_lpoly_data(coord.lpoly_data[j],coordnew.lpoly_data[j],coord.ngrid)
        test_lpoly_data(coord.element_data[j].lpoly_data,coordnew.element_data[j].lpoly_data,coord.ngrid)
        @test isapprox(coord.element_data[j].scale,coordnew.element_data[j].scale,atol=2.0e-15)
        @test isapprox(coord.element_data[j].shift,coordnew.element_data[j].shift,atol=2.0e-15)
    end

    return nothing
end

function test_lpoly_data(lpoly_data,lpoly_data_new,ngrid)
    @test isapprox(lpoly_data.x_nodes,
                   lpoly_data_new.x_nodes,
                    atol=2.0e-15)
    for i in 1:ngrid
        @test isapprox(lpoly_data.lpoly_data[i].other_nodes,
                lpoly_data_new.lpoly_data[i].other_nodes,
                atol=2.0e-15)
        @test isapprox(lpoly_data.lpoly_data[i].other_nodes_derivative,
                lpoly_data_new.lpoly_data[i].other_nodes_derivative,
                atol=2.0e-15)
        @test isapprox(lpoly_data.lpoly_data[i].one_over_denominator,
                lpoly_data_new.lpoly_data[i].one_over_denominator,
                atol=2.0e-15)
    end
    return nothing
end
ngrid = 5
nelement_global_vperp = 2
Lvperp = 3.0
element_spacing_option="uniform"

nelement_global_vpa = 4
Lvpa = 6.0
# create the coordinate structs
vperp = finite_element_coordinate("vperp", scalar_coordinate_inputs(ngrid,
                            nelement_global_vperp,
                            Lvperp),
                            element_spacing_option=element_spacing_option)
vpa = finite_element_coordinate("vpa", scalar_coordinate_inputs(ngrid,
                            nelement_global_vpa,
                            Lvpa),
                            element_spacing_option=element_spacing_option)


vperpnew = finite_element_coordinate("vperp",vperp.element_data)
vpanew = finite_element_coordinate("vpa",vpa.element_data)


@testset "coordinate creation" begin
    println("coordinate definition tests")
    @testset "vpa coordinate creation" begin
        println("    - test vpa coordinate creation")
        test_coordinates(vpa,vpanew)
    end
    @testset "vperp coordinate creation" begin
        println("    - test vperp coordinate creation")
        test_coordinates(vperp,vperpnew)
    end
end