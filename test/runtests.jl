module FokkerPlanckTests

using Test: @testset, @test
function runtests()
    @testset "FokkerPlanck tests" begin
        include(joinpath(@__DIR__, "calculus_tests.jl"))
        include(joinpath(@__DIR__, "velocity_integral_tests.jl"))
        include(joinpath(@__DIR__, "Interface.jl"))
        include(joinpath(@__DIR__, "fokker_planck_tests.jl"))
    end
end

end # FokkerPlanckTests

using .FokkerPlanckTests

FokkerPlanckTests.runtests()
