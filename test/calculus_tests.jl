module CalculusTests

using Test: @testset, @test
using StableRNGs
using FokkerPlanck.coordinates: finite_element_coordinate, first_derivative!, scalar_coordinate_inputs
using FokkerPlanck.calculus: integral
using LinearAlgebra: mul!, ldiv!

"""
Pass this function to the `norm` argument of `isapprox()` to test the maximum error
between two arrays.
"""
maxabs_norm(x) = maximum(abs.(x))

function runtests()
    @testset "calculus" begin
        println("calculus tests")
        @testset "fundamental theorem of calculus" begin
            @testset "$ngrid $nelement" for
                    ngrid ∈ (5,6,7,8,9,10), nelement ∈ (1, 2, 3, 4, 5)

                # define inputs needed for the test
                etol = 1.0e-14
                element_spacing_option = "uniform"
                L = 6.0
                bc = "none"
                # create the coordinate struct 'x'
                x = finite_element_coordinate("coord", scalar_coordinate_inputs(ngrid,
                                     nelement, L),
                                     bc=bc,
                                     element_spacing_option=element_spacing_option)
                # create array for the function f(x) to be differentiated/integrated
                f = Array{Float64,1}(undef, x.n)
                # create array for the derivative df/dx
                df = Array{Float64,1}(undef, x.n)
                # initialize f
                for ix ∈ 1:x.n
                    f[ix] = ( (cospi(2.0*x.grid[ix]/x.L)+sinpi(2.0*x.grid[ix]/x.L))
                              * exp(-x.grid[ix]^2) )
                end
                # differentiate f
                first_derivative!(df, f, x)
                # integrate df/dx
                intdf = integral(df, x.wgts)

                ## open ascii file to which test info will be written
                #io = open("tests.txt","w")
                #for ix ∈ 1:x.n
                #    println(io, "x: ", x.grid[ix], " f: ", f[ix], "  df: ", df[ix], " intdf: ", intdf)
                #end

                # Test that error intdf is less than the specified error tolerance etol
                @test abs(intdf) < etol
            end
        end

        rng = StableRNG(42)

        @testset "GaussLegendre pseudospectral derivatives (4 argument), testing exact polynomials" verbose=false begin
            @testset "$nelement $ngrid" for bc ∈ ("constant", "zero"),
                    nelement ∈ (1:5), ngrid ∈ (3:17)
                    
                # define inputs needed for the test
                L = 1.0
                bc = "constant"
                # create the coordinate struct 'x'
                x = finite_element_coordinate("coord", scalar_coordinate_inputs(ngrid,
                                      nelement, L),
                                      bc=bc,
                                      element_spacing_option="uniform")
                # test polynomials up to order ngrid-1
                for n ∈ 0:ngrid-1
                    # create array for the function f(x) to be differentiated/integrated
                    f = Array{Float64,1}(undef, x.n)
                    # create array for the derivative df/dx and the expected result
                    df = similar(f)
                    expected_df = similar(f)
                    # initialize f and expected df
                    f[:] .= randn(rng)
                    expected_df .= 0.0
                    for p ∈ 1:n
                        coefficient = randn(rng)
                        @. f += coefficient * x.grid ^ p
                        @. expected_df += coefficient * p * x.grid ^ (p - 1)
                    end
                    # differentiate f
                    first_derivative!(df, f, x)

                    # Note the error we might expect for a p=32 polynomial is probably
                    # something like p*(round-off) for x^p (?) so error on expected_df would
                    # be p*p*(round-off), or plausibly 1024*(round-off), so tolerance of
                    # 2e-11 isn't unreasonable.
                    @test isapprox(df, expected_df, rtol=2.0e-11, atol=6.0e-12,
                                   norm=maxabs_norm)
                end
            end
        end
    end        
end

end # CalculusTests


using .CalculusTests

CalculusTests.runtests()
