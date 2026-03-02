"""
Module for functions used
in calculating the integrals and doing
the numerical differentiation for
the implementation of the
the full-F Fokker-Planck collision operator.
"""
module fokker_planck_calculus

export assemble_explicit_collision_operator_rhs_serial!
export calculate_rosenbluth_potential_boundary_data!
export calculate_rosenbluth_potential_boundary_data_multipole!
export calculate_rosenbluth_potential_boundary_data_delta_f_multipole!
export FokkerPlanckArraysDirectIntegration
export FokkerPlanckWeakformArrays
export FokkerPlanckBackwardEulerData
export enforce_vpavperp_BCs!
export calculate_rosenbluth_potentials_via_elliptic_solve!
export calculate_rosenbluth_potentials_via_analytical_Maxwellian!
export calculate_test_particle_preconditioner!
export advance_linearised_test_particle_collisions!
export density_conserving_correction!, conserving_corrections!
export SpeciesData, calculate_cross_species_rosenbluth_potential_sums!
export calculate_rosenbluth_potential_boundary_data_exact!,
    calculate_analytical_Maxwellian_multipole_expansion_moments!
export test_rosenbluth_potential_boundary_data
export interpolate_2D_vspace!
export matrix_inverse
export multi_species_operator_type, single_assembly_per_species, repeat_assembly_per_species
export calculate_collision_moments!
export fokker_planck_collision_operator_solve!
export FixedBackgroundPlasmaInput,
    SlowingDownSourceInput,
    slowing_down_source!, slowing_down_sink!,
    add_slowing_down_source!
export convert_rosenbluth_potentials_from_source_to_other_grid!, DeltaFMultipoleMoments
export zero_boundary_condition, natural_boundary_condition
using ..velocity_moments: get_density, get_upar, get_pressure, get_ppar, get_pperp, get_qpar, get_rmom, get_nm_moment
using ..fokker_planck_test: F_Maxwellian, G_Maxwellian, H_Maxwellian, dHdvpa_Maxwellian, dHdvperp_Maxwellian
using ..fokker_planck_test: d2Gdvpa2_Maxwellian, d2Gdvperp2_Maxwellian, d2Gdvperpdvpa_Maxwellian, dGdvperp_Maxwellian
using Dates
using SpecialFunctions: ellipk, ellipe
using SparseArrays: sparse, AbstractSparseArray
using SuiteSparse
using LinearAlgebra: ldiv!, mul!, LU, ldiv, lu, lu!
using FastGaussQuadrature
using LagrangePolynomials: lagrange_poly, LagrangePolyData, lagrange_poly_derivative
using FiniteElementMatrices: lagrange_x,
                             d_lagrange_dx,
                             finite_element_matrix,
                             ElementCoordinates
using FiniteElementAssembly: first_derivative!, FiniteElementCoordinate, ScalarCoordinateInputs,
        BoundaryConditionType, NaturalBC, DirichletBC, assemble_operator, integral, get_ielement,
        value_in_coordinate_domain

using JacobianFreeNewtonKrylov: nl_solver_info
"""
Options for selecting which boundary data calculation to use
"""
@enum boundary_data_type begin
    direct_integration
    multipole_expansion
    delta_f_multipole
end
export boundary_data_type
export direct_integration
export multipole_expansion
export delta_f_multipole

"""
Option for selecting conservative terms strategy
"""
@enum multi_species_operator_type begin
    single_assembly_per_species
    repeat_assembly_per_species
end

"""
variables for boundary conditions imposed
on the evolved solution.
"""
# boundary condition imposed by structure of FE matrices
natural_boundary_condition=NaturalBC()
# Zero boundary conditions are imposed on the FE matrices and the solution
zero_boundary_condition=DirichletBC()

"""
Struct of dummy arrays and precalculated coefficients
for the Fokker-Planck collision operator when the
Rosenbluth potentials are computed everywhere in `(vpa,vperp)`
by direct integration. Used for testing.
"""
struct FokkerPlanckArraysDirectIntegration
    G0_weights::Array{Float64,4}
    G1_weights::Array{Float64,4}
    H0_weights::Array{Float64,4}
    H1_weights::Array{Float64,4}
    H2_weights::Array{Float64,4}
    H3_weights::Array{Float64,4}
    dfdvpa::Array{Float64,2}
    d2fdvperpdvpa::Array{Float64,2}
    dfdvperp::Array{Float64,2}
    """
    Function that initialises the arrays needed to calculate the Rosenbluth potentials
    by direct integration. As this function is only supported to keep the testing
    of the direct integration method, the struct created here does not contain
    all of the arrays necessary to compute the weak-form operator. This functionality
    could be ported if necessary.
    """
    function FokkerPlanckArraysDirectIntegration(vperp::FiniteElementCoordinate,
                                                        vpa::FiniteElementCoordinate;
                                                        print_to_screen=false::Bool)

        G0_weights = Array{Float64}(undef,vpa.n,vperp.n,vpa.n,vperp.n)
        G1_weights = Array{Float64}(undef,vpa.n,vperp.n,vpa.n,vperp.n)
        H0_weights = Array{Float64}(undef,vpa.n,vperp.n,vpa.n,vperp.n)
        H1_weights = Array{Float64}(undef,vpa.n,vperp.n,vpa.n,vperp.n)
        H2_weights = Array{Float64}(undef,vpa.n,vperp.n,vpa.n,vperp.n)
        H3_weights = Array{Float64}(undef,vpa.n,vperp.n,vpa.n,vperp.n)

        dfdvpa = Array{Float64}(undef,vpa.n,vperp.n)
        d2fdvperpdvpa = Array{Float64}(undef,vpa.n,vperp.n)
        dfdvperp = Array{Float64}(undef,vpa.n,vperp.n)
        if vperp.n > 1
            init_Rosenbluth_potential_integration_weights!(G0_weights, G1_weights, H0_weights, H1_weights,
                                            H2_weights, H3_weights, vperp, vpa, print_to_screen=print_to_screen)
        end
        return new(G0_weights, G1_weights, H0_weights,
                H1_weights, H2_weights, H3_weights,
                dfdvpa, d2fdvperpdvpa, dfdvperp)
    end
end

"""
Struct to contain the integration weights for the boundary points
in the `(vpa,vperp)` domain.
"""
struct BoundaryIntegrationWeights
    lower_vpa_boundary::Array{Float64,3}
    upper_vpa_boundary::Array{Float64,3}
    upper_vperp_boundary::Array{Float64,3}
    """
    Function to allocate a `BoundaryIntegrationWeights`.
    """
    function BoundaryIntegrationWeights(vpa::FiniteElementCoordinate,
                                                vperp::FiniteElementCoordinate)
        nvpa = vpa.n
        nvperp = vperp.n
        lower_vpa_boundary = Array{Float64}(undef,nvpa,nvperp,nvperp)
        upper_vpa_boundary = Array{Float64}(undef,nvpa,nvperp,nvperp)
        upper_vperp_boundary = Array{Float64}(undef,nvpa,nvperp,nvpa)
        return new(lower_vpa_boundary,
                upper_vpa_boundary, upper_vperp_boundary)
    end
end

"""
Struct used for storing the integration weights for the
boundary of the velocity space domain in `(vpa,vperp)` coordinates.
"""
struct FokkerPlanckBoundaryIntegration
    G0_weights::BoundaryIntegrationWeights
    G1_weights::BoundaryIntegrationWeights
    H0_weights::BoundaryIntegrationWeights
    H1_weights::BoundaryIntegrationWeights
    H2_weights::BoundaryIntegrationWeights
    H3_weights::BoundaryIntegrationWeights
    dfdvpa::Array{Float64,2}
    d2fdvperpdvpa::Array{Float64,2}
    dfdvperp::Array{Float64,2}
    """
    Function to allocate at `FokkerPlanckBoundaryIntegration`.
    """
    function FokkerPlanckBoundaryIntegration(vpa::FiniteElementCoordinate,
                                                vperp::FiniteElementCoordinate)
        G0_weights = BoundaryIntegrationWeights(vpa,vperp)
        G1_weights = BoundaryIntegrationWeights(vpa,vperp)
        H0_weights = BoundaryIntegrationWeights(vpa,vperp)
        H1_weights = BoundaryIntegrationWeights(vpa,vperp)
        H2_weights = BoundaryIntegrationWeights(vpa,vperp)
        H3_weights = BoundaryIntegrationWeights(vpa,vperp)

        nvpa = vpa.n
        nvperp = vperp.n
        dfdvpa = Array{Float64}(undef,nvpa,nvperp)
        d2fdvperpdvpa = Array{Float64}(undef,nvpa,nvperp)
        dfdvperp = Array{Float64}(undef,nvpa,nvperp)
        return new(G0_weights,G1_weights,H0_weights,
            H1_weights,H2_weights,H3_weights,
            dfdvpa,d2fdvperpdvpa,dfdvperp)
    end
end

"""
Struct to store the `(vpa,vperp)` boundary data for an
individual Rosenbluth potential.
"""
struct VpaVperpBoundaryData
    lower_boundary_vpa::Array{Float64,1}
    upper_boundary_vpa::Array{Float64,1}
    upper_boundary_vperp::Array{Float64,1}
    """
    Function to allocate an instance of `VpaVperpBoundaryData`.
    """
    #function allocate_boundary_data(vpa::FiniteElementCoordinate,
    function VpaVperpBoundaryData(vpa::FiniteElementCoordinate,
                                vperp::FiniteElementCoordinate)
        lower_boundary_vpa = Array{Float64}(undef,vperp.n)
        upper_boundary_vpa = Array{Float64}(undef,vperp.n)
        upper_boundary_vperp = Array{Float64}(undef,vpa.n)
        return new(lower_boundary_vpa,
                upper_boundary_vpa,upper_boundary_vperp)
    end
end

"""
Struct to store the boundary data for all of the
Rosenbluth potentials required for the calculation.
"""
struct RosenbluthPotentialBoundaryData
    H_data::VpaVperpBoundaryData
    dHdvpa_data::VpaVperpBoundaryData
    dHdvperp_data::VpaVperpBoundaryData
    G_data::VpaVperpBoundaryData
    dGdvperp_data::VpaVperpBoundaryData
    d2Gdvperp2_data::VpaVperpBoundaryData
    d2Gdvperpdvpa_data::VpaVperpBoundaryData
    d2Gdvpa2_data::VpaVperpBoundaryData
    """
    Function to allocate an instance of `RosenbluthPotentialBoundaryData`.
    """
    function RosenbluthPotentialBoundaryData(vpa::FiniteElementCoordinate,
                                                vperp::FiniteElementCoordinate)
        H_data = VpaVperpBoundaryData(vpa,vperp)
        dHdvpa_data = VpaVperpBoundaryData(vpa,vperp)
        dHdvperp_data = VpaVperpBoundaryData(vpa,vperp)
        G_data = VpaVperpBoundaryData(vpa,vperp)
        dGdvperp_data = VpaVperpBoundaryData(vpa,vperp)
        d2Gdvperp2_data = VpaVperpBoundaryData(vpa,vperp)
        d2Gdvperpdvpa_data = VpaVperpBoundaryData(vpa,vperp)
        d2Gdvpa2_data = VpaVperpBoundaryData(vpa,vperp)
        return new(H_data,dHdvpa_data,
            dHdvperp_data,G_data,dGdvperp_data,d2Gdvperp2_data,
            d2Gdvperpdvpa_data,d2Gdvpa2_data)
    end
end

"""
Struct to store the elemental nonlinear stiffness matrices used
to express the finite-element weak form of the collision
operator. The arrays are indexed so that the contraction
in the assembly step is carried out over the fastest
accessed indices, i.e., for `YYNperp[1,i,j,k,iel]`, we contract
over `i` and `j` to give data for the field position index `k`,
all for the 1D element indexed by `iel`.
"""
struct CollisionOperatorArrays
    # let phi_j(vperp) be the jth Lagrange basis function,
    # and phi'_j(vperp) the first derivative of the Lagrange basis function
    # on the iel^th element. Then, the arrays are defined as follows.
    # YYNperp[1,i,j,k,iel] = \int phi_i(vperp) phi_j(vperp) phi_k(vperp) vperp d vperp
    # YYNperp[2,i,j,k,iel] = \int phi_i(vperp) phi_j(vperp) phi'_k(vperp) vperp d vperp
    # YYNperp[3,i,j,k,iel] = \int phi_i(vperp) phi'_j(vperp) phi'_k(vperp) vperp d vperp
    # YYNperp[4,i,j,k,iel] = \int phi_i(vperp) phi'_j(vperp) phi_k(vperp) vperp d vperp
    YYNperp::Array{Float64,5}
    # MNperp[i,j,iel] = \int phi_i(vperp) phi_j(vperp) d vperp
    MNperp::Array{Float64,3}
    # MMperp[i,j,iel] = \int phi_i(vperp) phi_j(vperp) vperp d vperp
    MMperp::Array{Float64,3}
    # MRperp[i,j,iel] = \int phi_i(vperp) phi_j(vperp) vperp^2 d vperp
    MRperp::Array{Float64,3}
    # PQperp[i,j,iel] = \int phi'_i(vperp) phi_j(vperp) d vperp
    PQperp::Array{Float64,3}
    # PPperp[i,j,iel] = \int phi'_i(vperp) phi_j(vperp) vperp d vperp
    PPperp::Array{Float64,3}
    # PUperp[i,j,iel] = \int phi'_i(vperp) phi_j(vperp) vperp^2 d vperp
    PUperp::Array{Float64,3}
    # KKperp[i,j,iel] = -\int phi'_i(vperp) phi'_j(vperp) vperp d vperp
    KKperp::Array{Float64,3}
    # KJperp[i,j,iel] = -\int phi'_i(vperp) phi'_j(vperp) vperp^2 d vperp
    KJperp::Array{Float64,3}
    # KKperp[i,j,iel] = -\int phi'_i(vperp) phi'_j(vperp) vperp d vperp
    KKperp_with_BC_terms::Array{Float64,3}
    # YYNpar[1,i,j,k,iel] = \int phi_i(vpa) phi_j(vpa) phi_k(vpa) d vpa
    # YYNpar[2,i,j,k,iel] = \int phi_i(vpa) phi_j(vpa) phi'_k(vpa) d vpa
    # YYNpar[3,i,j,k,iel] = \int phi_i(vpa) phi'_j(vpa) phi'_k(vpa) d vpa
    # YYNpar[4,i,j,k,iel] = \int phi_i(vpa) phi'_j(vpa) phi_k(vpa) d vpa
    YYNpar::Array{Float64,5}
    # MMpar[i,j,iel] = \int phi_i(vpa) phi_j(vpa) d vpa
    MMpar::Array{Float64,3}
    # MRpar[i,j,iel] = \int phi_i(vpa) phi_j(vpa) vpa d vpa
    MRpar::Array{Float64,3}
    # PPpar[i,j,iel] = \int phi_i(vpa) phi'_j(vpa) d vpa
    PPpar::Array{Float64,3}
    # PUpar[i,j,iel] = \int phi'_i(vpa) phi_j(vpa) vpa d vpa
    PUpar::Array{Float64,3}
    # KKpar[i,j,iel] = -\int phi'_i(vpa) phi'_j(vpa) d vpa
    KKpar::Array{Float64,3}
    # KKpar[i,j,iel] = -\int phi'_i(vpa) phi'_j(vpa) d vpa
    KKpar_with_BC_terms::Array{Float64,3}
    """
    Function to allocate an instance of `CollisionOperatorArrays`.
    Definitions of these nonlinear stiffness matrices can be found in
    the type definition of `CollisionOperatorArrays`.
    """
    function CollisionOperatorArrays(vpa::FiniteElementCoordinate,vperp::FiniteElementCoordinate)
        YYNperp = Array{Float64,5}(undef,4,vperp.ngrid,vperp.ngrid,vperp.ngrid,vperp.nelement)
        MNperp = Array{Float64,3}(undef,vperp.ngrid,vperp.ngrid,vperp.nelement)
        MMperp = Array{Float64,3}(undef,vperp.ngrid,vperp.ngrid,vperp.nelement)
        MRperp = Array{Float64,3}(undef,vperp.ngrid,vperp.ngrid,vperp.nelement)
        PQperp = Array{Float64,3}(undef,vperp.ngrid,vperp.ngrid,vperp.nelement)
        PPperp = Array{Float64,3}(undef,vperp.ngrid,vperp.ngrid,vperp.nelement)
        PUperp = Array{Float64,3}(undef,vperp.ngrid,vperp.ngrid,vperp.nelement)
        KKperp = Array{Float64,3}(undef,vperp.ngrid,vperp.ngrid,vperp.nelement)
        KJperp = Array{Float64,3}(undef,vperp.ngrid,vperp.ngrid,vperp.nelement)
        KKperp_with_BC_terms = Array{Float64,3}(undef,vperp.ngrid,vperp.ngrid,vperp.nelement)

        YYNpar = Array{Float64,5}(undef,4,vpa.ngrid,vpa.ngrid,vpa.ngrid,vpa.nelement)
        MMpar = Array{Float64,3}(undef,vpa.ngrid,vpa.ngrid,vpa.nelement)
        MRpar = Array{Float64,3}(undef,vpa.ngrid,vpa.ngrid,vpa.nelement)
        PPpar = Array{Float64,3}(undef,vpa.ngrid,vpa.ngrid,vpa.nelement)
        PUpar = Array{Float64,3}(undef,vpa.ngrid,vpa.ngrid,vpa.nelement)
        KKpar = Array{Float64,3}(undef,vpa.ngrid,vpa.ngrid,vpa.nelement)
        KKpar_with_BC_terms = Array{Float64,3}(undef,vpa.ngrid,vpa.ngrid,vpa.nelement)


        for ielement_vperp in 1:vperp.nelement
            element_data = vperp.element_data[ielement_vperp]
            @views YYNperp[1,:,:,:,ielement_vperp] = finite_element_matrix(lagrange_x,lagrange_x,lagrange_x,1,element_data)
            @views YYNperp[2,:,:,:,ielement_vperp] = finite_element_matrix(lagrange_x,lagrange_x,d_lagrange_dx,1,element_data)
            @views YYNperp[3,:,:,:,ielement_vperp] = finite_element_matrix(lagrange_x,d_lagrange_dx,d_lagrange_dx,1,element_data)
            @views YYNperp[4,:,:,:,ielement_vperp] = finite_element_matrix(lagrange_x,d_lagrange_dx,lagrange_x,1,element_data)
            @views MNperp[:,:,ielement_vperp] = finite_element_matrix(lagrange_x,lagrange_x,0,element_data)
            @views MMperp[:,:,ielement_vperp] = finite_element_matrix(lagrange_x,lagrange_x,1,element_data)
            @views MRperp[:,:,ielement_vperp] = finite_element_matrix(lagrange_x,lagrange_x,2,element_data)
            @views PQperp[:,:,ielement_vperp] = finite_element_matrix(d_lagrange_dx,lagrange_x,0,element_data)
            @views PPperp[:,:,ielement_vperp] = finite_element_matrix(d_lagrange_dx,lagrange_x,1,element_data)
            @views PUperp[:,:,ielement_vperp] = finite_element_matrix(d_lagrange_dx,lagrange_x,2,element_data)
            @views KKperp[:,:,ielement_vperp] = -finite_element_matrix(d_lagrange_dx,d_lagrange_dx,1,element_data)
            @views KJperp[:,:,ielement_vperp] = -finite_element_matrix(d_lagrange_dx,d_lagrange_dx,2,element_data)
            @views KKperp_with_BC_terms[:,:,ielement_vperp] .= KKperp[:,:,ielement_vperp]
            if ielement_vperp == vperp.nelement
                imax = vperp.imax[ielement_vperp]
                vperp_max = vperp.grid[imax]
                ivperp = vperp.ngrid
                for jvperp in 1:vperp.ngrid
                    KKperp_with_BC_terms[jvperp,ivperp,ielement_vperp] += lagrange_poly_derivative(vperp.lpoly_data[ielement_vperp].lpoly_data[jvperp],vperp_max)*vperp_max
                end
            end
        end
        for ielement_vpa in 1:vpa.nelement
            element_data = vpa.element_data[ielement_vpa]
            @views YYNpar[1,:,:,:,ielement_vpa] = finite_element_matrix(lagrange_x,lagrange_x,lagrange_x,0,element_data)
            @views YYNpar[2,:,:,:,ielement_vpa] = finite_element_matrix(lagrange_x,lagrange_x,d_lagrange_dx,0,element_data)
            @views YYNpar[3,:,:,:,ielement_vpa] = finite_element_matrix(lagrange_x,d_lagrange_dx,d_lagrange_dx,0,element_data)
            @views YYNpar[4,:,:,:,ielement_vpa] = finite_element_matrix(lagrange_x,d_lagrange_dx,lagrange_x,0,element_data)
            @views MMpar[:,:,ielement_vpa] = finite_element_matrix(lagrange_x,lagrange_x,0,element_data)
            @views MRpar[:,:,ielement_vpa] = finite_element_matrix(lagrange_x,lagrange_x,1,element_data)
            @views PPpar[:,:,ielement_vpa] = finite_element_matrix(d_lagrange_dx,lagrange_x,0,element_data)
            @views PUpar[:,:,ielement_vpa] = finite_element_matrix(d_lagrange_dx,lagrange_x,1,element_data)
            @views KKpar[:,:,ielement_vpa] = -finite_element_matrix(d_lagrange_dx,d_lagrange_dx,0,element_data)
            @views KKpar_with_BC_terms[:,:,ielement_vpa] .= KKpar[:,:,ielement_vpa]
            if ielement_vpa == 1
                imin = vpa.imin[ielement_vpa]
                vpa_min = vpa.grid[imin]
                ivpa = 1
                for jvpa in 1:vpa.ngrid
                    KKpar_with_BC_terms[jvpa,ivpa,ielement_vpa] -= lagrange_poly_derivative(vpa.lpoly_data[ielement_vpa].lpoly_data[jvpa],vpa_min)
                end
            end
            if ielement_vpa == vpa.nelement
                imax = vpa.imax[ielement_vpa]
                vpa_max = vpa.grid[imax]
                ivpa = vpa.ngrid
                for jvpa in 1:vpa.ngrid
                    KKpar_with_BC_terms[jvpa,ivpa,ielement_vpa] += lagrange_poly_derivative(vpa.lpoly_data[ielement_vpa].lpoly_data[jvpa],vpa_max)
                end
            end
        end

        return new(YYNperp,
                MNperp,MMperp,MRperp,
                PQperp,PPperp,PUperp,
                KKperp,KJperp,KKperp_with_BC_terms,
                YYNpar,
                MMpar,MRpar,PPpar,PUpar,
                KKpar,KKpar_with_BC_terms)
    end
end


struct AssembledSparseMatrixOperators{TFloat<:Float64,
        TSparseMatrix <: AbstractSparseArray{TFloat,Int64,2}}
    # assembled 2D weak-form matrices
    MM2D_sparse::TSparseMatrix
    KKpar2D_sparse::TSparseMatrix
    KKperp2D_sparse::TSparseMatrix
    KKpar2D_with_BC_terms_sparse::TSparseMatrix
    KKperp2D_with_BC_terms_sparse::TSparseMatrix
    LP2D_sparse::TSparseMatrix
    LV2D_sparse::TSparseMatrix
    LB2D_sparse::TSparseMatrix
    PUperp2D_sparse::TSparseMatrix
    PPparPUperp2D_sparse::TSparseMatrix
    PPpar2D_sparse::TSparseMatrix
    MMparMNperp2D_sparse::TSparseMatrix
    KPperp2D_sparse::TSparseMatrix
    # lu decomposition objects
    lu_obj_MM::SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}
    lu_obj_LP::SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}
    lu_obj_LV::SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}
    lu_obj_LB::SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}
    # dummy arrays for elliptic solvers
    S_dummy::Array{Float64,2}
    Q_dummy::Array{Float64,2}
    rhsvpavperp::Array{Float64,2}
    """
    Function to contruct the global sparse matrices used to solve
    the elliptic PDEs for the Rosenbluth potentials. The function
    Uses a sparse matrix construction method from FiniteElementAssembly.
    """
    function AssembledSparseMatrixOperators(vpa::FiniteElementCoordinate,
                                            vperp::FiniteElementCoordinate,
                                            YY_arrays::CollisionOperatorArrays;
                                            print_to_screen=true)
        if print_to_screen
            println("begin elliptic operator assignment   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
        # expand some variables to make weak form expressions more concise
        MMperp = YY_arrays.MMperp
        MRperp = YY_arrays.MRperp
        MNperp = YY_arrays.MNperp
        KKperp = YY_arrays.KKperp
        KJperp = YY_arrays.KJperp
        PPperp = YY_arrays.PPperp
        PUperp = YY_arrays.PUperp
        KKperp_with_BC_terms = YY_arrays.KKperp_with_BC_terms
        MMpar = YY_arrays.MMpar
        KKpar = YY_arrays.KKpar
        KKpar_with_BC_terms = YY_arrays.KKpar_with_BC_terms
        PPpar = YY_arrays.PPpar
        # weak forms followed by sparse matrix constructor pairs
        function MM2D_weak_form(jvpa, ivpa, ielement_vpa, jvperp, ivperp, ielement_vperp)
            return MMpar[jvpa,ivpa,ielement_vpa]*MMperp[jvperp,ivperp,ielement_vperp]
        end
        function KKpar2D_weak_form(jvpa, ivpa, ielement_vpa, jvperp, ivperp, ielement_vperp)
            return KKpar[jvpa,ivpa,ielement_vpa]*MMperp[jvperp,ivperp,ielement_vperp]
        end
        function KKpar2D_BC_terms_weak_form(jvpa, ivpa, ielement_vpa, jvperp, ivperp, ielement_vperp)
            return KKpar_with_BC_terms[jvpa,ivpa,ielement_vpa]*MMperp[jvperp,ivperp,ielement_vperp]
        end
        function KKperp2D_weak_form(jvpa, ivpa, ielement_vpa, jvperp, ivperp, ielement_vperp)
            return MMpar[jvpa,ivpa,ielement_vpa]*KKperp[jvperp,ivperp,ielement_vperp]
        end
        function KKperp2D_BC_terms_weak_form(jvpa, ivpa, ielement_vpa, jvperp, ivperp, ielement_vperp)
            return MMpar[jvpa,ivpa,ielement_vpa]*KKperp_with_BC_terms[jvperp,ivperp,ielement_vperp]
        end
        function KPperp2D_weak_form(jvpa, ivpa, ielement_vpa, jvperp, ivperp, ielement_vperp)
            return (MMpar[jvpa,ivpa,ielement_vpa]*
                    (KJperp[jvperp,ivperp,ielement_vperp] -
                    2.0*PPperp[jvperp,ivperp,ielement_vperp] -
                    2.0*MNperp[jvperp,ivperp,ielement_vperp]))
        end
        function PUperp2D_weak_form(jvpa, ivpa, ielement_vpa, jvperp, ivperp, ielement_vperp)
            return MMpar[jvpa,ivpa,ielement_vpa]*PUperp[jvperp,ivperp,ielement_vperp]
        end
        function PPpar2D_weak_form(jvpa, ivpa, ielement_vpa, jvperp, ivperp, ielement_vperp)
            return PPpar[jvpa,ivpa,ielement_vpa]*MMperp[jvperp,ivperp,ielement_vperp]
        end
        function PPparPUperp2D_weak_form(jvpa, ivpa, ielement_vpa, jvperp, ivperp, ielement_vperp)
            return PPpar[jvpa,ivpa,ielement_vpa]*PUperp[jvperp,ivperp,ielement_vperp]
        end
        function MMparMNperp2D_weak_form(jvpa, ivpa, ielement_vpa, jvperp, ivperp, ielement_vperp)
            return MMpar[jvpa,ivpa,ielement_vpa]*MNperp[jvperp,ivperp,ielement_vperp]
        end
        function LP2D_weak_form(jvpa, ivpa, ielement_vpa, jvperp, ivperp, ielement_vperp)
            return (KKpar[jvpa,ivpa,ielement_vpa]*MMperp[jvperp,ivperp,ielement_vperp] +
                    MMpar[jvpa,ivpa,ielement_vpa]*KKperp[jvperp,ivperp,ielement_vperp])
        end
        function LV2D_weak_form(jvpa, ivpa, ielement_vpa,
                            jvperp, ivperp, ielement_vperp)
            return (KKpar[jvpa,ivpa,ielement_vpa]*
                    MRperp[jvperp,ivperp,ielement_vperp] +
                    MMpar[jvpa,ivpa,ielement_vpa]*
                    (KJperp[jvperp,ivperp,ielement_vperp] -
                    PPperp[jvperp,ivperp,ielement_vperp] -
                    MNperp[jvperp,ivperp,ielement_vperp]))
        end
        function LB2D_weak_form(jvpa, ivpa, ielement_vpa,
                            jvperp, ivperp, ielement_vperp)
            return (KKpar[jvpa,ivpa,ielement_vpa]*
                    MRperp[jvperp,ivperp,ielement_vperp] +
                    MMpar[jvpa,ivpa,ielement_vpa]*
                    (KJperp[jvperp,ivperp,ielement_vperp] -
                    PPperp[jvperp,ivperp,ielement_vperp] -
                    4.0*MNperp[jvperp,ivperp,ielement_vperp]))
        end
        MM2D_sparse = assemble_operator(MM2D_weak_form, vpa, vperp, NaturalBC(), NaturalBC())
        MMparMNperp2D_sparse = assemble_operator(MMparMNperp2D_weak_form, vpa, vperp, NaturalBC(), NaturalBC())
        KKpar2D_sparse = assemble_operator(KKpar2D_weak_form, vpa, vperp, NaturalBC(), NaturalBC())
        KKpar2D_with_BC_terms_sparse = assemble_operator(KKpar2D_BC_terms_weak_form, vpa, vperp, NaturalBC(), NaturalBC())
        KKperp2D_sparse = assemble_operator(KKperp2D_weak_form, vpa, vperp, NaturalBC(), NaturalBC())
        KKperp2D_with_BC_terms_sparse = assemble_operator(KKperp2D_BC_terms_weak_form, vpa, vperp, NaturalBC(), NaturalBC())
        PUperp2D_sparse = assemble_operator(PUperp2D_weak_form, vpa, vperp, NaturalBC(), NaturalBC())
        PPparPUperp2D_sparse = assemble_operator(PPparPUperp2D_weak_form, vpa, vperp, NaturalBC(), NaturalBC())
        PPpar2D_sparse = assemble_operator(PPpar2D_weak_form, vpa, vperp, NaturalBC(), NaturalBC())
        KPperp2D_sparse = assemble_operator(KPperp2D_weak_form, vpa, vperp, NaturalBC(), NaturalBC())
        LP2D_sparse = assemble_operator(LP2D_weak_form, vpa, vperp, DirichletBC(), DirichletBC())
        LV2D_sparse = assemble_operator(LV2D_weak_form, vpa, vperp, DirichletBC(), DirichletBC())
        LB2D_sparse = assemble_operator(LB2D_weak_form, vpa, vperp, DirichletBC(), DirichletBC())
        if print_to_screen
            println("finished elliptic operator constructor assignment   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
        # lu objects for the assembled operators
        lu_obj_MM = lu(MM2D_sparse)
        lu_obj_LP = lu(LP2D_sparse)
        lu_obj_LV = lu(LV2D_sparse)
        lu_obj_LB = lu(LB2D_sparse)
        # dummy arrays for elliptic solvers
        S_dummy = Array{Float64}(undef,vpa.n,vperp.n)
        Q_dummy = Array{Float64}(undef,vpa.n,vperp.n)
        rhsvpavperp = Array{Float64}(undef,vpa.n,vperp.n)
        return new{Float64,typeof(MM2D_sparse)}(MM2D_sparse,KKpar2D_sparse,KKperp2D_sparse,
                    KKpar2D_with_BC_terms_sparse,KKperp2D_with_BC_terms_sparse,
                    LP2D_sparse,LV2D_sparse,LB2D_sparse,PUperp2D_sparse,PPparPUperp2D_sparse,
                    PPpar2D_sparse,MMparMNperp2D_sparse,KPperp2D_sparse,
                    lu_obj_MM,lu_obj_LP,lu_obj_LV,lu_obj_LB,
                    S_dummy, Q_dummy, rhsvpavperp)
    end
end

struct DeltaFMultipoleMoments
    Inm_vec::Vector{Float64}
    Maxwellian_moments::Vector{Float64}
end

struct RosenbluthPotentialData
    # dummy arrays for storing Rosenbluth potentials (vpa,vperp)
    GG::Array{Float64,2}
    HH::Array{Float64,2}
    dHdvpa::Array{Float64,2}
    dHdvperp::Array{Float64,2}
    dGdvperp::Array{Float64,2}
    d2Gdvperp2::Array{Float64,2}
    d2Gdvpa2::Array{Float64,2}
    d2Gdvperpdvpa::Array{Float64,2}
    # the moments required to reconstuct the multipole expansion
    multipole_expansion_moments::Union{DeltaFMultipoleMoments,Vector{Float64},Nothing}
    function RosenbluthPotentialData(vpa::FiniteElementCoordinate,
                                vperp::FiniteElementCoordinate,
                                boundary_data_option::boundary_data_type)
        GG = Array{Float64}(undef,vpa.n,vperp.n)
        HH = Array{Float64}(undef,vpa.n,vperp.n)
        dHdvpa = Array{Float64}(undef,vpa.n,vperp.n)
        dHdvperp = Array{Float64}(undef,vpa.n,vperp.n)
        dGdvperp = Array{Float64}(undef,vpa.n,vperp.n)
        d2Gdvperp2 = Array{Float64}(undef,vpa.n,vperp.n)
        d2Gdvpa2 = Array{Float64}(undef,vpa.n,vperp.n)
        d2Gdvperpdvpa = Array{Float64}(undef,vpa.n,vperp.n)
        Inm_vec = Array{Float64}(undef,25)
        Maxwellian_moments = Array{Float64}(undef,3)
        if boundary_data_option == delta_f_multipole
            multipole_expansion_moments = DeltaFMultipoleMoments(Inm_vec,Maxwellian_moments)
        elseif boundary_data_option == multipole_expansion
            multipole_expansion_moments = Inm_vec
        else # option to ensure that `boundary_data_option == direct_integration`
             # can be distinguished from the other options at compile time
            multipole_expansion_moments = nothing
        end
        return new(GG, HH, dHdvpa, dHdvperp, dGdvperp, d2Gdvperp2, d2Gdvpa2, d2Gdvperpdvpa,
                    multipole_expansion_moments)
    end
end

struct FokkerPlanckRosenbluthPotentialSolverData
    # boundary weights (Green's function) data
    bwgt::FokkerPlanckBoundaryIntegration
    # dummy arrays for boundary data calculation
    rpbd::RosenbluthPotentialBoundaryData
    # option for boundary data calculation
    boundary_data_option::boundary_data_type
    # assembled 2D weak-form matrices
    matrix_operators::AssembledSparseMatrixOperators
    function FokkerPlanckRosenbluthPotentialSolverData(vpa::FiniteElementCoordinate,
                                                    vperp::FiniteElementCoordinate,
                                                    YY_arrays::CollisionOperatorArrays,
                                                    boundary_data_option::boundary_data_type;
                                                    print_to_screen=true)
        bwgt = FokkerPlanckBoundaryIntegration(vpa,vperp)
        if vperp.n > 1 && boundary_data_option == direct_integration
            init_Rosenbluth_potential_boundary_integration_weights!(bwgt.G0_weights, bwgt.G1_weights, bwgt.H0_weights, bwgt.H1_weights,
                                            bwgt.H2_weights, bwgt.H3_weights, vpa, vperp, print_to_screen=print_to_screen)
        end
        rpbd = RosenbluthPotentialBoundaryData(vpa,vperp)
        matrix_operators = AssembledSparseMatrixOperators(vpa,vperp,YY_arrays,print_to_screen=print_to_screen)
        return new(bwgt, rpbd, boundary_data_option, matrix_operators)
    end
end

"""
Information about each species
"""
struct SpeciesData
    # number of species
    n::Int64
    # mass of each species
    mass::Vector{Float64}
    # charge number of each species
    zeds::Vector{Float64}
    # reference speed for each species, given with respect to the species-independent c_ref
    c0ref::Vector{Float64}
    # reference parallel velocity for each species, given with respect to the species-independent c_ref
    u0ref::Vector{Float64}
    # reference density for each species, , given with respect to the species-independent n_ref
    n0ref::Vector{Float64}
    """
    Internal constructor for `SpeciesData`.
    """
    function SpeciesData(mass::Vector{Float64},zeds::Vector{Float64})
        # constructor where no reference information is supplied
        nspecies = length(zeds)
        c0ref = ones(nspecies)
        u0ref = zeros(nspecies)
        n0ref = ones(nspecies)
        return SpeciesData(mass,zeds,c0ref,u0ref,n0ref)
    end
    function SpeciesData(mass::Vector{Float64},zeds::Vector{Float64},
        c0ref::Vector{Float64}, u0ref::Vector{Float64}, n0ref::Vector{Float64})
        # number of species
        nspecies = length(zeds)
        # check inputs are consistent
        @boundscheck nspecies == length(mass) || throw(BoundsError(mass))
        @boundscheck nspecies == length(c0ref) || throw(BoundsError(c0ref))
        @boundscheck nspecies == length(u0ref) || throw(BoundsError(u0ref))
        @boundscheck nspecies == length(n0ref) || throw(BoundsError(n0ref))
        for is in 1:nspecies
            # check mass positive and > 0
            if mass[is] < 1.0e-12
                error("ERROR: mass[$is] < 1.0e-12")
            end
            # check ref density positive and > 0
            if n0ref[is] < 1.0e-12
                error("ERROR: n0ref[$is] < 1.0e-12")
            end
            # check ref speed positive and > 0
            if c0ref[is] < 1.0e-12
                error("ERROR: c0ref[$is] < 1.0e-12")
            end
        end
        return new(nspecies,mass,zeds,c0ref,u0ref,n0ref)
    end
end


"""
"""
struct PdfMoments
    density::Vector{Float64}
    upar::Vector{Float64}
    vth::Vector{Float64}
end
"""
"""
struct FixedBackgroundPlasmaInput{Tpdf <: Union{PdfMoments,AbstractArray{Float64,3}}}
    species::SpeciesData
    pdf::Tpdf
    function FixedBackgroundPlasmaInput(mass::Vector{Float64},
                                    zeds::Vector{Float64},
                                    density::Vector{Float64},
                                    upar::Vector{Float64},
                                    vth::Vector{Float64})
        pdf = PdfMoments(density,upar,vth)
        return FixedBackgroundPlasmaInput(mass,zeds,pdf)
    end
    function FixedBackgroundPlasmaInput(mass::Vector{Float64},
                zeds::Vector{Float64},
                pdf::Union{PdfMoments,Tpdf}
                ) where Tpdf <: AbstractArray{Float64,3}
        species = SpeciesData(mass,zeds)
        return new{typeof(pdf)}(species,pdf)
    end
end
"""
"""
struct FixedBackgroundPlasmaData
    species::SpeciesData
    rosenbluth_potentials_s::Vector{RosenbluthPotentialData}
    # internal constructor for Maxwellian background plasma species
    function FixedBackgroundPlasmaData(fixed_background_plasma_in::FixedBackgroundPlasmaInput,
            vpa::FiniteElementCoordinate, vperp::FiniteElementCoordinate,
            fprp_solver_data::FokkerPlanckRosenbluthPotentialSolverData)
        species = fixed_background_plasma_in.species
        pdf = fixed_background_plasma_in.pdf
        return FixedBackgroundPlasmaData(species, pdf, vpa, vperp, fprp_solver_data)
    end
    function FixedBackgroundPlasmaData(species::SpeciesData, pdf::PdfMoments,
                vpa::FiniteElementCoordinate, vperp::FiniteElementCoordinate,
                fprp_solver_data::FokkerPlanckRosenbluthPotentialSolverData)
        density = pdf.density
        upar = pdf.upar
        vth = pdf.vth
        @boundscheck species.n == length(density) || throw(BoundsError(density))
        @boundscheck species.n == length(upar) || throw(BoundsError(upar))
        @boundscheck species.n == length(vth) || throw(BoundsError(vth))
        rosenbluth_potentials_s = Vector{RosenbluthPotentialData}(undef,species.n)
        for is in 1:species.n
            rosenbluth_potentials_s[is] = RosenbluthPotentialData(vpa,vperp,
                                            fprp_solver_data.boundary_data_option)
        end
        for is in 1:species.n
            calculate_rosenbluth_potentials_via_analytical_Maxwellian!(
                rosenbluth_potentials_s[is],density[is],upar[is],vth[is],vpa,vperp)
        end
        return new(species, rosenbluth_potentials_s)
    end
    # internal constructor for non-Maxwellian background plasma species
    function FixedBackgroundPlasmaData(species::SpeciesData, pdf::Tpdf,
                vpa::FiniteElementCoordinate, vperp::FiniteElementCoordinate,
                fprp_solver_data::FokkerPlanckRosenbluthPotentialSolverData
                ) where Tpdf <: AbstractArray{Float64,3}
        @boundscheck (species.n == size(pdf,3) && vperp.n == size(pdf,2) && vpa.n == size(pdf,1)) || throw(BoundsError(pdf))
        rosenbluth_potentials_s = Vector{RosenbluthPotentialData}(undef,species.n)
        for is in 1:species.n
            rosenbluth_potentials_s[is] = RosenbluthPotentialData(vpa,vperp,
                                            fprp_solver_data.boundary_data_option)
        end
        for is in 1:species.n
            @views calculate_rosenbluth_potentials_via_elliptic_solve!(rosenbluth_potentials_s[is],pdf[:,:,is],
             vpa,vperp,fprp_solver_data,
             algebraic_solve_for_d2Gdvperp2=false,calculate_GG=false,
             calculate_dGdvperp=false)
        end
        return new(species, rosenbluth_potentials_s)
    end
end
"""
Struct of dummy arrays and precalculated coefficients
for the finite-element weak-form Fokker-Planck collision operator.
"""
struct FokkerPlanckWeakformArrays
    # vpa (v||) coordinate struct
    vpa::FiniteElementCoordinate
    # vperp coordinate struct
    vperp::FiniteElementCoordinate
    # species information
    species::SpeciesData
    # data for Rosenbluth potential elliptic solver
    fprp_solver_data::FokkerPlanckRosenbluthPotentialSolverData
    # elemental matrices for the assembly of C[Fs,Fsp]
    YY_arrays::CollisionOperatorArrays
    # dummy arrays for storing Rosenbluth potentials (vpa,vperp,species)
    rosenbluth_potentials_s::Vector{RosenbluthPotentialData}
    # dummy arrays for storing Rosenbluth potentials (vpa,vperp)
    rosenbluth_potentials::RosenbluthPotentialData
    rosenbluth_potentials_buffer::RosenbluthPotentialData
    # collision operator moment arrays
    delta_n_sp_s::Array{Float64,2}
    delta_m_sp_s::Array{Float64,2}
    delta_p_sp_s::Array{Float64,2}
    density::Array{Float64,1}
    upar::Array{Float64,1}
    pressure::Array{Float64,1}
    ppar::Array{Float64,1}
    qpar::Array{Float64,1}
    rmom::Array{Float64,1}
    delta_n::Array{Float64,1}
    delta_P::Array{Float64,1}
    delta_E::Array{Float64,1}
    # conserving correction coefficients
    correction_coeffs_z::Array{Float64,3}
    # dummy array for end-of-step corrections
    delta_pdf::Array{Float64,3}
    # option to control which numerical error corrections method to use
    multi_species_operator_option::multi_species_operator_type
    # data needed to include contributions from unevolved plasma species
    fixed_background_plasma::Union{FixedBackgroundPlasmaData,Nothing}
    """
    Function that initialises the arrays needed for Fokker Planck collisions
    using numerical integration to compute the Rosenbluth potentials only
    at the boundary and using an elliptic solve to obtain the potentials
    in the rest of the velocity space domain.
    """
    function FokkerPlanckWeakformArrays(vpa::FiniteElementCoordinate,
                vperp::FiniteElementCoordinate,
                species::SpeciesData,
                boundary_data_option::boundary_data_type;
                multi_species_operator_option=single_assembly_per_species::multi_species_operator_type,
                print_to_screen=true::Bool,
                fixed_background_plasma_in=nothing::Union{FixedBackgroundPlasmaInput,Nothing})
        YY_arrays = CollisionOperatorArrays(vpa,vperp)
        fprp_solver_data = FokkerPlanckRosenbluthPotentialSolverData(vpa,vperp,
                                YY_arrays,boundary_data_option,print_to_screen=print_to_screen)
        nvpa, nvperp, nspecies = vpa.n, vperp.n, species.n
        rosenbluth_potentials_s = Vector{RosenbluthPotentialData}(undef,nspecies)
        for is in 1:nspecies
            rosenbluth_potentials_s[is] = RosenbluthPotentialData(vpa,vperp,boundary_data_option)
        end
        rosenbluth_potentials = RosenbluthPotentialData(vpa,vperp,boundary_data_option)
        rosenbluth_potentials_buffer = RosenbluthPotentialData(vpa,vperp,boundary_data_option)

        # multi-species conserving corrections
        delta_n_sp_s = Array{Float64}(undef,nspecies,nspecies)
        delta_m_sp_s = Array{Float64}(undef,nspecies,nspecies)
        delta_p_sp_s = Array{Float64}(undef,nspecies,nspecies)
        density = Array{Float64}(undef,nspecies)
        upar = Array{Float64}(undef,nspecies)
        pressure = Array{Float64}(undef,nspecies)
        ppar = Array{Float64}(undef,nspecies)
        qpar = Array{Float64}(undef,nspecies)
        rmom = Array{Float64}(undef,nspecies)
        delta_n = Array{Float64}(undef,nspecies)
        delta_P = Array{Float64}(undef,nspecies)
        delta_E = Array{Float64}(undef,nspecies)
        correction_coeffs_z = Array{Float64}(undef,3,nspecies,nspecies)
        delta_pdf = Array{Float64}(undef,nvpa,nvperp,nspecies)
        # Rosenbluth potentials and species info for unevolved species
        if typeof(fixed_background_plasma_in) == Nothing
            fixed_background_plasma = nothing
        else
            fixed_background_plasma = FixedBackgroundPlasmaData(fixed_background_plasma_in,
                                            vpa,vperp,fprp_solver_data)
        end
        return new(vpa, vperp, species, fprp_solver_data, YY_arrays,
                    rosenbluth_potentials_s, rosenbluth_potentials,
                    rosenbluth_potentials_buffer,
                    delta_n_sp_s, delta_m_sp_s, delta_p_sp_s,
                    density, upar, pressure, ppar, qpar, rmom,
                    delta_n, delta_P, delta_E, correction_coeffs_z, delta_pdf,
                    multi_species_operator_option, fixed_background_plasma)
    end
end

struct SlowingDownSourceInput
    source_rate::Array{Float64,1}
    source_vth::Array{Float64,1}
    source_v0::Array{Float64,1}
    sink_rate::Array{Float64,1}
    sink_vth::Float64
    constant_sink::Bool
end
struct SlowingDownSourceData
    # arrays needed for slowing down sources and sinks
    sink_func::Array{Float64,2}
    source_input::SlowingDownSourceInput
    function SlowingDownSourceData(vpa::FiniteElementCoordinate,
            vperp::FiniteElementCoordinate, species::SpeciesData,
            source_input::SlowingDownSourceInput)
        source_rate = source_input.source_rate
        source_vth = source_input.source_vth
        source_v0 = source_input.source_v0
        sink_rate = source_input.sink_rate
        sink_vth = source_input.sink_vth
        sink_func = Array{Float64}(undef,vpa.n,vperp.n)
        if source_input.constant_sink
            @. sink_func = 1.0
        else
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    v2 = vpa.grid[ivpa]^2 + vperp.grid[ivperp]^2
                    norm = 1.0/((sqrt(pi)*sink_vth)^3)
                    sink_func[ivpa,ivperp] = norm*exp(-v2/(sink_vth^2))
                end
            end
        end
        @boundscheck species.n == size(source_rate,1) || throw(BoundsError(source_rate))
        @boundscheck species.n == size(source_vth,1) || throw(BoundsError(source_vth))
        @boundscheck species.n == size(source_v0,1) || throw(BoundsError(source_v0))
        @boundscheck species.n == size(sink_rate,1) || throw(BoundsError(sink_rate))
        return new(sink_func, source_input)
    end
end
function slowing_down_source!(source::Tpdf,
    fkpl_arrays::FokkerPlanckWeakformArrays, source_data::SlowingDownSourceData
    ) where Tpdf <: AbstractArray{Float64,3}
    # extract variables for source
    source_rate = source_data.source_input.source_rate
    source_vth = source_data.source_input.source_vth
    source_v0 = source_data.source_input.source_v0
    vpa = fkpl_arrays.vpa
    vperp = fkpl_arrays.vperp
    species = fkpl_arrays.species
    dummy_vpavperp = fkpl_arrays.fprp_solver_data.matrix_operators.S_dummy
    # source of alphas
    for is in 1:species.n
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                v2 = vperp.grid[ivperp]^2 + vpa.grid[ivpa]^2
                fac = 0.25/((source_vth[is]*source_v0[is])^2)
                dummy_vpavperp[ivpa,ivperp] = exp(-fac*(v2 - source_v0[is]^2)^2 )
            end
        end
        normfac = get_density(dummy_vpavperp, vpa, vperp)
        @. source[:,:,is] += source_rate[is]*dummy_vpavperp/normfac
    end
    return nothing
end
function slowing_down_sink!(source::Tpdf1, pdf::Tpdf2,
    fkpl_arrays::FokkerPlanckWeakformArrays,
    source_data::SlowingDownSourceData
    ) where {Tpdf1 <: AbstractArray{Float64,3}, Tpdf2 <: AbstractArray{Float64,3}}
    # extract variables for sink of alphas
    vpa = fkpl_arrays.vpa
    vperp = fkpl_arrays.vperp
    species = fkpl_arrays.species
    YY_arrays = fkpl_arrays.YY_arrays
    lu_obj_MM = fkpl_arrays.fprp_solver_data.matrix_operators.lu_obj_MM
    S_dummy = fkpl_arrays.fprp_solver_data.matrix_operators.S_dummy
    rhsvpavperp = fkpl_arrays.fprp_solver_data.matrix_operators.rhsvpavperp
    rhsc = vec(rhsvpavperp)
    sc = vec(S_dummy)
    sink_rate = source_data.source_input.sink_rate
    sink_func = source_data.sink_func
    # sink of alphas
    for is in 1:species.n
        @. rhsc = 0.0
        @views pdfs = pdf[:,:,is]
        for ielement_vperp in 1:vperp.nelement
            @views YYNperp = YY_arrays.YYNperp[:,:,:,:,ielement_vperp]
            @views vperp_igrid_full = vperp.igrid_full[:,ielement_vperp]
            imin_vperp, imax_vperp = vperp_igrid_full[1], vperp_igrid_full[vperp.ngrid]
            for ielement_vpa in 1:vpa.nelement
                @views YYNpar = YY_arrays.YYNpar[:,:,:,:,ielement_vpa]
                @views vpa_igrid_full = vpa.igrid_full[:,ielement_vpa]
                imin_vpa, imax_vpa = vpa_igrid_full[1], vpa_igrid_full[vpa.ngrid]
                @views sink_func_local = sink_func[imin_vpa:imax_vpa,imin_vperp:imax_vperp]
                @views pdfs_local = pdfs[imin_vpa:imax_vpa,imin_vperp:imax_vperp]
                for ivperp_local in 1:vperp.ngrid
                    ivperp_global = vperp_igrid_full[ivperp_local]
                    for ivpa_local in 1:vpa.ngrid
                        ivpa_global = vpa_igrid_full[ivpa_local]
                        # global compound index
                        ic_global = ic_func(ivpa_global,ivperp_global,vpa.n)
                        # carry out the matrix sum on each 2D element
                        result = 0.0
                        for jvperpp_local in 1:vperp.ngrid
                            for kvperpp_local in 1:vperp.ngrid
                                @views YYNperp_kji = YYNperp[:,kvperpp_local,jvperpp_local,ivperp_local]
                                for jvpap_local in 1:vpa.ngrid
                                    pdfjj = pdfs_local[jvpap_local,jvperpp_local]
                                    for kvpap_local in 1:vpa.ngrid
                                        @views YYNpar_kji = YYNpar[:,kvpap_local,jvpap_local,ivpa_local]
                                        result += pdfjj*YYNperp_kji[1]*YYNpar_kji[1]*sink_func_local[kvpap_local,kvperpp_local]
                                    end
                                end
                            end
                        end
                        rhsc[ic_global] += result
                    end
                end
            end
        end
        # invert mass matrix to get collocation point values
        ldiv!(sc, lu_obj_MM, rhsc)
        @views @. source[:,:,is] -= sink_rate[is]*S_dummy
    end
    return nothing
end
function slowing_down_source!(source::Tpdf,
    fkpl_arrays::FokkerPlanckWeakformArrays, source_data::Nothing
    ) where Tpdf <: AbstractArray{Float64,3}
    # do nothing
    return nothing
end
function slowing_down_sink!(source::Tpdf1, pdf::Tpdf2,
    fkpl_arrays::FokkerPlanckWeakformArrays, source_data::Nothing
    ) where {Tpdf1 <: AbstractArray{Float64,3}, Tpdf2 <: AbstractArray{Float64,3}}
    # do nothing
    return nothing
end
function add_slowing_down_source!(Fnew::Tpdf1,
    Fdummy::Tpdf2, fkpl_arrays::FokkerPlanckWeakformArrays,
    source_data::SlowingDownSourceData, delta_t::Float64
    ) where {Tpdf1 <: AbstractArray{Float64,3}, Tpdf2 <: AbstractArray{Float64,3}}
    @. Fdummy = 0.0 # set dummy to zero before adding source
    slowing_down_source!(Fdummy, fkpl_arrays, source_data)
    @. Fnew += delta_t*Fdummy
    return nothing
end
function add_slowing_down_source!(Fnew::Tpdf1,
    Fdummy::Tpdf2, fkpl_arrays::FokkerPlanckWeakformArrays,
    source_data::Nothing, delta_t::Float64
    ) where {Tpdf1 <: AbstractArray{Float64,3}, Tpdf2 <: AbstractArray{Float64,3}}
    # do nothing
    return nothing
end

struct FokkerPlanckBackwardEulerData{TFloat<:Float64,
        TSparseMatrix <: AbstractSparseArray{TFloat,Int64,2}}
    # arrays for storing collision operator computed
    # when iterating in the backward Euler step
    CCs::Array{Float64,3}
    # array for storing the sources to preserve collision diagnostics
    source::Array{Float64,3}
    # matrices for storing preconditioner
    # based on I - dt * C[delta F, F]
    CC2D_sparse::TSparseMatrix
    lu_obj_CC2D::SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}
    lu_objs_CC2D::Array{SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64},1}
    # dummy arrays for Jacobian-Free-Newton-Krylov solver
    # multispecies dummy arrays
    nl_solver_data_s::nl_solver_info{Array{Float64,2},Array{Float64,4},Array{Float64,1}}
    Fs_new::Array{Float64,3}
    Fs_residual::Array{Float64,3}
    Fs_delta_x::Array{Float64,3}
    Fs_rhs_delta::Array{Float64,3}
    Fsv::Array{Float64,3}
    Fsw::Array{Float64,3}
    fp_operator::FokkerPlanckWeakformArrays
    source_data::Union{SlowingDownSourceData,Nothing}
    # constructor with interface and JFNK optional arguments
    """
    Wrapper function to provide the interface for initialising the
    Fokker Planck operator arrays and operators. We require that
    the inputs are provided with the types
    ```
        inputs =  ScalarCoordinateInputs(ngrid, nelement, L)
    ```
    or
    ```
        inputs = Array{ElementCoordinates,1}(undef, nelement)
    ```
    where the former type is defined in `FokkerPlanck.coordinates`
    and the latterr is defined in `FiniteElementMatrices`.
    """
    function FokkerPlanckBackwardEulerData(
        mass::Vector{Float64},
        zeds::Vector{Float64},
        c0refs::Vector{Float64},
        u0refs::Vector{Float64},
        n0refs::Vector{Float64},
        inputs_vpa::Union{ScalarCoordinateInputs,Array{ElementCoordinates,1}},
        inputs_vperp::Union{ScalarCoordinateInputs,Array{ElementCoordinates,1}};
        bc_vpa=natural_boundary_condition::BoundaryConditionType,
        bc_vperp=natural_boundary_condition::BoundaryConditionType,
        boundary_data_option=multipole_expansion::boundary_data_type,
        multi_species_operator_option=single_assembly_per_species::multi_species_operator_type,
        nl_solver_atol=1.0e-10::Float64,
        nl_solver_rtol=0.0::Float64,
        nl_solver_nonlinear_max_iterations=20::Int64,
        print_to_screen=true::Bool,
        fixed_background_plasma_in=nothing::Union{FixedBackgroundPlasmaInput,Nothing},
        source_data_in=nothing::Union{SlowingDownSourceInput,Nothing})
        # create the coordinate structs from the input data
        vperp = FiniteElementCoordinate("vperp", inputs_vperp,
                                    bc=bc_vperp, weight_function=((vperp)-> 2.0*pi*vperp))
        vpa = FiniteElementCoordinate("vpa", inputs_vpa,
                                    bc=bc_vpa)
        species = SpeciesData(mass,zeds,c0refs,u0refs,n0refs)
        # use constructor function for FokkerPlanckWeakformArrays
        return FokkerPlanckBackwardEulerData(vpa,vperp,species,
                    boundary_data_option,multi_species_operator_option,
                    nl_solver_atol,nl_solver_rtol,nl_solver_nonlinear_max_iterations,
                    print_to_screen,fixed_background_plasma_in,source_data_in)
    end
    # constructor without optional arguments
    function FokkerPlanckBackwardEulerData(vpa::FiniteElementCoordinate,
                                    vperp::FiniteElementCoordinate,
                                    species::SpeciesData,
                                    boundary_data_option::boundary_data_type,
                                    multi_species_operator_option::multi_species_operator_type,
                                    nl_solver_atol::Float64,
                                    nl_solver_rtol::Float64,
                                    nl_solver_nonlinear_max_iterations::Int64,
                                    print_to_screen::Bool,
                                    fixed_background_plasma_in::Union{FixedBackgroundPlasmaInput,Nothing},
                                    source_data_in::Union{SlowingDownSourceInput,Nothing})
        nvpa, nvperp, nspecies = vpa.n, vperp.n, species.n
        # collision operator arrays for intermediate results
        CCs = Array{Float64}(undef,nvpa,nvperp,nspecies)
        source = Array{Float64}(undef,nvpa,nvperp,nspecies)
        # preconditioner matrix
        CC2D_sparse, lu_obj_CC2D = allocate_preconditioner_matrix(vpa,vperp)
        lu_objs_CC2D = Array{SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64},1}(undef,nspecies)
        for is in 1:nspecies
            lu_objs_CC2D[is] = lu_obj_CC2D
        end
        # dummy arrays for JFNK
        nl_solver_data_s = nl_solver_info((species=species,vperp=vperp,vpa=vpa);
                                        atol=nl_solver_atol,
                                        rtol=nl_solver_rtol,
                                        nonlinear_max_iterations=nl_solver_nonlinear_max_iterations)
        Fs_new = Array{Float64}(undef,nvpa,nvperp,nspecies)
        Fs_residual = Array{Float64}(undef,nvpa,nvperp,nspecies)
        Fs_delta_x = Array{Float64}(undef,nvpa,nvperp,nspecies)
        Fs_rhs_delta = Array{Float64}(undef,nvpa,nvperp,nspecies)
        Fsv = Array{Float64}(undef,nvpa,nvperp,nspecies)
        Fsw = Array{Float64}(undef,nvpa,nvperp,nspecies)
        # data for the FP operators
        fp_operator = FokkerPlanckWeakformArrays(vpa,vperp,species,
                                                boundary_data_option;
                                                multi_species_operator_option=multi_species_operator_option,
                                                print_to_screen=print_to_screen,
                                                fixed_background_plasma_in=fixed_background_plasma_in)
        if typeof(source_data_in) == SlowingDownSourceInput
            source_data = SlowingDownSourceData(vpa,vperp,species,source_data_in)
        else
            source_data = nothing
        end
        return new{Float64,typeof(CC2D_sparse)}(CCs,source,
            CC2D_sparse,lu_obj_CC2D,
            lu_objs_CC2D,
            nl_solver_data_s,
            Fs_new,Fs_residual,Fs_delta_x,Fs_rhs_delta,Fsv,Fsw,
            fp_operator,source_data)
    end
end

"""
Function that precomputes the required integration weights in the whole of
`(vpa,vperp)` for the direct integration method of computing the Rosenbluth potentials.
"""
function init_Rosenbluth_potential_integration_weights!(G0_weights::Twgts,G1_weights::Twgts,H0_weights::Twgts,
                H1_weights::Twgts,H2_weights::Twgts,H3_weights::Twgts,
                vperp::FiniteElementCoordinate,vpa::FiniteElementCoordinate;
                print_to_screen=true) where Twgts <: AbstractArray{Float64,4}

    x_vpa, w_vpa, x_vperp, w_vperp, x_legendre, w_legendre, x_laguerre, w_laguerre = setup_basic_quadratures(vpa,vperp,print_to_screen=print_to_screen)

    if print_to_screen
        println("beginning weights calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    end

    # precalculated weights, integrating over Lagrange polynomials
    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                #limits where checks required to determine which divergence-safe grid is needed
                igrid_vpa, ielement_vpa, ielement_vpa_low, ielement_vpa_hi, igrid_vperp, ielement_vperp, ielement_vperp_low, ielement_vperp_hi = get_element_limit_indices(ivpa,ivperp,vpa,vperp)

                vperp_val = vperp.grid[ivperp]
                vpa_val = vpa.grid[ivpa]
                for ivperpp in 1:vperp.n
                    for ivpap in 1:vpa.n
                        G0_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0
                        G1_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0
                        # G2_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0
                        # G3_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0
                        H0_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0
                        H1_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0
                        H2_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0
                        H3_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0
                        #@. n_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0
                    end
                end
                # loop over elements and grid points within elements on primed coordinate
                @views loop_over_vperp_vpa_elements!(G0_weights[:,:,ivpa,ivperp],G1_weights[:,:,ivpa,ivperp],
                        H0_weights[:,:,ivpa,ivperp],H1_weights[:,:,ivpa,ivperp],
                        H2_weights[:,:,ivpa,ivperp],H3_weights[:,:,ivpa,ivperp],
                        vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                        vperp,ielement_vperp_low,ielement_vperp_hi, # info about primed vperp grids
                        x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                        x_legendre,w_legendre,x_laguerre,w_laguerre,
                        igrid_vpa, igrid_vperp, vpa_val, vperp_val)
            end
        end
    end


    if print_to_screen
        println("finished weights calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    end
    return nothing
end

"""
Function for getting the basic quadratures used for the
numerical integration of the Lagrange polynomials and the
integration kernals.
"""
function setup_basic_quadratures(vpa::FiniteElementCoordinate,
            vperp::FiniteElementCoordinate;print_to_screen=true)
    if print_to_screen
        println("setting up GL quadrature   ", Dates.format(now(), dateformat"H:MM:SS"))
    end

    # get Gauss-Legendre points and weights on (-1,1)
    ngrid = max(vpa.ngrid,vperp.ngrid)
    nquad = 2*ngrid
    x_legendre, w_legendre = gausslegendre(nquad)
    #nlaguerre = min(9,nquad) # to prevent points to close to the boundaries
    nlaguerre = nquad
    x_laguerre, w_laguerre = gausslaguerre(nlaguerre)

    x_vpa, w_vpa = Array{Float64,1}(undef,4*nquad), Array{Float64,1}(undef,4*nquad)
    x_vperp, w_vperp = Array{Float64,1}(undef,4*nquad), Array{Float64,1}(undef,4*nquad)

    return x_vpa, w_vpa, x_vperp, w_vperp, x_legendre, w_legendre, x_laguerre, w_laguerre
end


"""
Function for getting the indices used to choose the integration quadrature.
"""
function get_element_limit_indices(ivpa::Int64,ivperp::Int64,
                            vpa::FiniteElementCoordinate,
                            vperp::FiniteElementCoordinate)
    #limits where checks required to determine which divergence-safe grid is needed
    igrid_vpa, ielement_vpa = vpa.igrid[ivpa], vpa.ielement[ivpa]
    ielement_vpa_low = ielement_vpa - ng_low(igrid_vpa,vpa.ngrid)*nel_low(ielement_vpa,vpa.nelement)
    ielement_vpa_hi = ielement_vpa + ng_hi(igrid_vpa,vpa.ngrid)*nel_hi(ielement_vpa,vpa.nelement)
    #println("igrid_vpa: ielement_vpa: ielement_vpa_low: ielement_vpa_hi:", igrid_vpa," ",ielement_vpa," ",ielement_vpa_low," ",ielement_vpa_hi)
    igrid_vperp, ielement_vperp = vperp.igrid[ivperp], vperp.ielement[ivperp]
    ielement_vperp_low = ielement_vperp - ng_low(igrid_vperp,vperp.ngrid)*nel_low(ielement_vperp,vperp.nelement)
    ielement_vperp_hi = ielement_vperp + ng_hi(igrid_vperp,vperp.ngrid)*nel_hi(ielement_vperp,vperp.nelement)
    #println("igrid_vperp: ielement_vperp: ielement_vperp_low: ielement_vperp_hi:", igrid_vperp," ",ielement_vperp," ",ielement_vperp_low," ",ielement_vperp_hi)
    return igrid_vpa, ielement_vpa, ielement_vpa_low, ielement_vpa_hi,
            igrid_vperp, ielement_vperp, ielement_vperp_low, ielement_vperp_hi
end

"""
Function that precomputes the required integration weights only along the velocity space boundaries.
Used as the default option as part of the strategy to compute the Rosenbluth potentials
at the boundaries with direct integration and in the rest of `(vpa,vperp)` by solving elliptic PDEs.
"""
function init_Rosenbluth_potential_boundary_integration_weights!(G0_weights::Twgts,
      G1_weights::Twgts,H0_weights::Twgts,H1_weights::Twgts,
      H2_weights::Twgts,H3_weights::Twgts,
      vpa::FiniteElementCoordinate,vperp::FiniteElementCoordinate;
      print_to_screen=true) where Twgts <: BoundaryIntegrationWeights

    x_vpa, w_vpa, x_vperp, w_vperp, x_legendre, w_legendre, x_laguerre, w_laguerre = setup_basic_quadratures(vpa,vperp,print_to_screen=print_to_screen)

    if print_to_screen
        println("beginning (boundary) weights calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    end

    # precalculate weights, integrating over Lagrange polynomials
    # first compute weights along lower vpa boundary
    ivpa = 1 # lower_vpa_boundary
    @inbounds for ivperp in 1:vperp.n
        #limits where checks required to determine which divergence-safe grid is needed
        igrid_vpa, ielement_vpa, ielement_vpa_low, ielement_vpa_hi, igrid_vperp, ielement_vperp, ielement_vperp_low, ielement_vperp_hi = get_element_limit_indices(ivpa,ivperp,vpa,vperp)

        vperp_val = vperp.grid[ivperp]
        vpa_val = vpa.grid[ivpa]
        for ivperpp in 1:vperp.n
            for ivpap in 1:vpa.n
                G0_weights.lower_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0
                G1_weights.lower_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0
                # G2_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0
                # G3_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0
                H0_weights.lower_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0
                H1_weights.lower_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0
                H2_weights.lower_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0
                H3_weights.lower_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0
                #@. n_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0
            end
        end
        # loop over elements and grid points within elements on primed coordinate
        @views loop_over_vperp_vpa_elements!(G0_weights.lower_vpa_boundary[:,:,ivperp],
                G1_weights.lower_vpa_boundary[:,:,ivperp],
                H0_weights.lower_vpa_boundary[:,:,ivperp],
                H1_weights.lower_vpa_boundary[:,:,ivperp],
                H2_weights.lower_vpa_boundary[:,:,ivperp],
                H3_weights.lower_vpa_boundary[:,:,ivperp],
                vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                vperp,ielement_vperp_low,ielement_vperp_hi, # info about primed vperp grids
                x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                x_legendre,w_legendre,x_laguerre,w_laguerre,
                igrid_vpa, igrid_vperp, vpa_val, vperp_val)
    end
    # second compute weights along upper vpa boundary
    ivpa = vpa.n # upper_vpa_boundary
    @inbounds for ivperp in 1:vperp.n
        #limits where checks required to determine which divergence-safe grid is needed
        igrid_vpa, ielement_vpa, ielement_vpa_low, ielement_vpa_hi, igrid_vperp, ielement_vperp, ielement_vperp_low, ielement_vperp_hi = get_element_limit_indices(ivpa,ivperp,vpa,vperp)

        vperp_val = vperp.grid[ivperp]
        vpa_val = vpa.grid[ivpa]
        for ivperpp in 1:vperp.n
            for ivpap in 1:vpa.n
                G0_weights.upper_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0
                G1_weights.upper_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0
                # G2_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0
                # G3_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0
                H0_weights.upper_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0
                H1_weights.upper_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0
                H2_weights.upper_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0
                H3_weights.upper_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0
                #@. n_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0
            end
        end
        # loop over elements and grid points within elements on primed coordinate
        @views loop_over_vperp_vpa_elements!(G0_weights.upper_vpa_boundary[:,:,ivperp],
                G1_weights.upper_vpa_boundary[:,:,ivperp],
                H0_weights.upper_vpa_boundary[:,:,ivperp],
                H1_weights.upper_vpa_boundary[:,:,ivperp],
                H2_weights.upper_vpa_boundary[:,:,ivperp],
                H3_weights.upper_vpa_boundary[:,:,ivperp],
                vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                vperp,ielement_vperp_low,ielement_vperp_hi, # info about primed vperp grids
                x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                x_legendre,w_legendre,x_laguerre,w_laguerre,
                igrid_vpa, igrid_vperp, vpa_val, vperp_val)
    end
    # finally compute weight along upper vperp boundary
    ivperp = vperp.n # upper_vperp_boundary
    @inbounds for ivpa in 1:vpa.n
        #limits where checks required to determine which divergence-safe grid is needed
        igrid_vpa, ielement_vpa, ielement_vpa_low, ielement_vpa_hi, igrid_vperp, ielement_vperp, ielement_vperp_low, ielement_vperp_hi = get_element_limit_indices(ivpa,ivperp,vpa,vperp)

        vperp_val = vperp.grid[ivperp]
        vpa_val = vpa.grid[ivpa]
        for ivperpp in 1:vperp.n
            for ivpap in 1:vpa.n
                G0_weights.upper_vperp_boundary[ivpap,ivperpp,ivpa] = 0.0
                G1_weights.upper_vperp_boundary[ivpap,ivperpp,ivpa] = 0.0
                # G2_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0
                # G3_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0
                H0_weights.upper_vperp_boundary[ivpap,ivperpp,ivpa] = 0.0
                H1_weights.upper_vperp_boundary[ivpap,ivperpp,ivpa] = 0.0
                H2_weights.upper_vperp_boundary[ivpap,ivperpp,ivpa] = 0.0
                H3_weights.upper_vperp_boundary[ivpap,ivperpp,ivpa] = 0.0
                #@. n_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0
            end
        end
        # loop over elements and grid points within elements on primed coordinate
        @views loop_over_vperp_vpa_elements!(G0_weights.upper_vperp_boundary[:,:,ivpa],
                G1_weights.upper_vperp_boundary[:,:,ivpa],
                H0_weights.upper_vperp_boundary[:,:,ivpa],
                H1_weights.upper_vperp_boundary[:,:,ivpa],
                H2_weights.upper_vperp_boundary[:,:,ivpa],
                H3_weights.upper_vperp_boundary[:,:,ivpa],
                vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                vperp,ielement_vperp_low,ielement_vperp_hi, # info about primed vperp grids
                x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                x_legendre,w_legendre,x_laguerre,w_laguerre,
                igrid_vpa, igrid_vperp, vpa_val, vperp_val)
    end
    # return the parallelisation status to serial
    if print_to_screen
        println("finished (boundary) weights calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    end
    return nothing
end

function get_imin_imax(coord::FiniteElementCoordinate,iel::Int64)
    j = iel
    if j > 1
        k = 1
    else
        k = 0
    end
    imin = coord.imin[j] - k
    imax = coord.imax[j]
    return imin, imax
end

function get_nodes(coord::FiniteElementCoordinate,iel::Int64)
    # get imin and imax of this element on full grid
    (imin, imax) = get_imin_imax(coord,iel)
    nodes = coord.grid[imin:imax]
    return nodes
end

"""
Function to get the local integration grid and quadrature weights
to integrate a 1D element in the 2D representation of the
velocity space distribution functions. This function assumes that
there is a divergence at the point `coord_val`, and splits the grid
and integration weights appropriately, using Gauss-Laguerre points
near the divergence and Gauss-Legendre points away from the divergence.
"""
function get_scaled_x_w_with_divergences!(x_scaled::Tx, w_scaled::Tx,
                            x_legendre::Tx, w_legendre::Tx,
                            x_laguerre::Tx, w_laguerre::Tx,
                            node_min::Float64, node_max::Float64, nodes::Tx,
                            igrid_coord::Int64, coord_val::Float64) where Tx <: AbstractArray{Float64,1}
    #println("nodes ",nodes)
    zero = 1.0e-10
    @. x_scaled = 0.0
    @. w_scaled = 0.0
    nnodes = size(nodes,1)
    nquad_legendre = size(x_legendre,1)
    nquad_laguerre = size(x_laguerre,1)
    # assume x_scaled, w_scaled are arrays of length 2*nquad
    # use only nquad points for most elements, but use 2*nquad for
    # elements with interior divergences
    #println("coord: ",coord_val," node_max: ",node_max," node_min: ",node_min)
    if abs(coord_val - node_max) < zero # divergence at upper endpoint
        node_cut = (nodes[nnodes-1] + nodes[nnodes])/2.0

        n = nquad_laguerre + nquad_legendre
        shift = 0.5*(node_min + node_cut)
        scale = 0.5*(node_cut - node_min)
        @. x_scaled[1:nquad_legendre] = scale*x_legendre + shift
        @. w_scaled[1:nquad_legendre] = scale*w_legendre

        @. x_scaled[1+nquad_legendre:n] = node_max + (node_cut - node_max)*exp(-x_laguerre)
        @. w_scaled[1+nquad_legendre:n] = (node_max - node_cut)*w_laguerre

        nquad_coord = n
        #println("upper divergence")
    elseif abs(coord_val - node_min) < zero # divergence at lower endpoint
        n = nquad_laguerre + nquad_legendre
        nquad = size(x_laguerre,1)
        node_cut = (nodes[1] + nodes[2])/2.0
        for j in 1:nquad_laguerre
            x_scaled[nquad_laguerre+1-j] = node_min + (node_cut - node_min)*exp(-x_laguerre[j])
            w_scaled[nquad_laguerre+1-j] = (node_cut - node_min)*w_laguerre[j]
        end
        shift = 0.5*(node_max + node_cut)
        scale = 0.5*(node_max - node_cut)
        @. x_scaled[1+nquad_laguerre:n] = scale*x_legendre + shift
        @. w_scaled[1+nquad_laguerre:n] = scale*w_legendre

        nquad_coord = n
        #println("lower divergence")
    else #if (coord_val - node_min)*(coord_val - node_max) < - zero # interior divergence
        #println(nodes[igrid_coord]," ", coord_val)
        n = 2*nquad_laguerre
        node_cut_high = (nodes[igrid_coord+1] + nodes[igrid_coord])/2.0
        if igrid_coord == 1
            # exception for vperp coordinate near orgin
            k = 0
            node_cut_low = node_min
            nquad_coord = nquad_legendre + 2*nquad_laguerre
        else
            # fill in lower Gauss-Legendre points
            node_cut_low = (nodes[igrid_coord-1]+nodes[igrid_coord])/2.0
            shift = 0.5*(node_cut_low + node_min)
            scale = 0.5*(node_cut_low - node_min)
            @. x_scaled[1:nquad_legendre] = scale*x_legendre + shift
            @. w_scaled[1:nquad_legendre] = scale*w_legendre
            k = nquad_legendre
            nquad_coord = 2*(nquad_laguerre + nquad_legendre)
        end
        # lower half of domain
        for j in 1:nquad_laguerre
            x_scaled[k+j] = coord_val + (node_cut_low - coord_val)*exp(-x_laguerre[j])
            w_scaled[k+j] = (coord_val - node_cut_low)*w_laguerre[j]
        end
        # upper half of domain
        for j in 1:nquad_laguerre
            x_scaled[k+n+1-j] = coord_val + (node_cut_high - coord_val)*exp(-x_laguerre[j])
            w_scaled[k+n+1-j] = (node_cut_high - coord_val)*w_laguerre[j]
        end
        # fill in upper Gauss-Legendre points
        shift = 0.5*(node_cut_high + node_max)
        scale = 0.5*(node_max - node_cut_high)
        @. x_scaled[k+n+1:nquad_coord] = scale*x_legendre + shift
        @. w_scaled[k+n+1:nquad_coord] = scale*w_legendre

        #println("intermediate divergence")
    #else # no divergences
    #    nquad = size(x_legendre,1)
    #    shift = 0.5*(node_min + node_max)
    #    scale = 0.5*(node_max - node_min)
    #    @. x_scaled[1:nquad] = scale*x_legendre + shift
    #    @. w_scaled[1:nquad] = scale*w_legendre
    #    #println("no divergence")
    #    nquad_coord = nquad
    end
    #println("x_scaled",x_scaled)
    #println("w_scaled",w_scaled)
    return nquad_coord
end

"""
Function to get the local grid and integration weights assuming
no divergences of the function on the 1D element. Gauss-Legendre
quadrature is used for the entire element.
"""
function get_scaled_x_w_no_divergences!(x_scaled::Tx, w_scaled::Tx,
                                x_legendre::Tx, w_legendre::Tx,
                                node_min::Float64, node_max::Float64) where Tx <: AbstractArray{Float64,1}
    @. x_scaled = 0.0
    @. w_scaled = 0.0
    #println("coord: ",coord_val," node_max: ",node_max," node_min: ",node_min)
    nquad = size(x_legendre,1)
    shift = 0.5*(node_min + node_max)
    scale = 0.5*(node_max - node_min)
    @. x_scaled[1:nquad] = scale*x_legendre + shift
    @. w_scaled[1:nquad] = scale*w_legendre
    #println("x_scaled",x_scaled)
    #println("w_scaled",w_scaled)
    return nquad
end

"""
Function returns `1` if `igrid = 1` or `0` if `1 < igrid <= ngrid`.
"""
function ng_low(igrid::Int64,ngrid::Int64)
    return floor(Int64, (ngrid - igrid)/(ngrid - 1))
end

"""
Function returns `1` if `igrid = ngrid` or `0` if `1 =< igrid < ngrid`.
"""
function ng_hi(igrid::Int64,ngrid::Int64)
    return floor(Int64, igrid/ngrid)
end

"""
Function returns `1` for `nelement >= ielement > 1`, `0` for `ielement = 1`.
"""
function nel_low(ielement::Int64,nelement::Int64)
    return floor(Int64, (ielement - 2 + nelement)/nelement)
end

"""
Function returns `1` for `nelement > ielement >= 1`, `0` for `ielement = nelement`.
"""
function nel_hi(ielement::Int64,nelement::Int64)
    return 1- floor(Int64, ielement/nelement)
end

"""
Base level function for computing the integration kernals for the Rosenbluth potential integration.
Note the definitions of `ellipe(m)` (\$E(m)\$) and `ellipk(m)` (\$K(m)\$).
`https://specialfunctions.juliamath.org/stable/functions_list/#SpecialFunctions.ellipe`
`https://specialfunctions.juliamath.org/stable/functions_list/#SpecialFunctions.ellipk`
```math
E(m) = \\int^{\\pi/2}_0 \\sqrt{ 1 - m \\sin^2(\\theta)} d \\theta
```
```math
K(m) = \\int^{\\pi/2}_0 \\frac{1}{\\sqrt{ 1 - m \\sin^2(\\theta)}} d \\theta
```
"""
function local_element_integration!(G0_weights::Twgts,G1_weights::Twgts,H0_weights::Twgts,
                            H1_weights::Twgts,H2_weights::Twgts,H3_weights::Twgts,
                            # info about primed vpa grids
                            nquad_vpa::Int64,ielement_vpa::Int64,vpa::FiniteElementCoordinate,
                            # info about primed vperp grids
                            nquad_vperp::Int64,ielement_vperp::Int64,vperp::FiniteElementCoordinate,
                            # points and weights for primed (source) grids
                            x_vpa::Tx, w_vpa::Tx, x_vperp::Tx, w_vperp::Tx,
                            # values and indices for unprimed (field) grids
                            vpa_val::Float64, vperp_val::Float64) where {Twgts <: AbstractArray{Float64,2}, Tx <: AbstractArray{Float64,1}}
    @inbounds begin
        vperp_lpoly_data = vperp.lpoly_data[ielement_vperp]
        vpa_lpoly_data = vpa.lpoly_data[ielement_vpa]
        for igrid_vperp in 1:vperp.ngrid
            igrid_vperp_lpoly_data = vperp_lpoly_data.lpoly_data[igrid_vperp]
            for igrid_vpa in 1:vpa.ngrid
                igrid_vpa_lpoly_data = vpa_lpoly_data.lpoly_data[igrid_vpa]
                # get grid index for point on full grid
                ivpap = vpa.igrid_full[igrid_vpa,ielement_vpa]
                ivperpp = vperp.igrid_full[igrid_vperp,ielement_vperp]
                # carry out integration over Lagrange polynomial at this node, on this element
                for kvperp in 1:nquad_vperp
                    for kvpa in 1:nquad_vpa
                        x_kvpa = x_vpa[kvpa]
                        x_kvperp = x_vperp[kvperp]
                        w_kvperp = w_vperp[kvperp]
                        w_kvpa = w_vpa[kvpa]
                        denom = (vpa_val - x_kvpa)^2 + (vperp_val + x_kvperp)^2
                        mm = min(4.0*vperp_val*x_kvperp/denom,1.0 - 1.0e-15)
                        #mm = 4.0*vperp_val*x_kvperp/denom/(1.0 + 10^-15)
                        #mm = 4.0*vperp_val*x_kvperp/denom
                        prefac = sqrt(denom)
                        ellipe_mm = ellipe(mm)
                        ellipk_mm = ellipk(mm)
                        #if mm_test > 1.0
                        #    println("mm: ",mm_test," ellipe: ",ellipe_mm," ellipk: ",ellipk_mm)
                        #end
                        G_elliptic_integral_factor = 2.0*ellipe_mm*prefac*sqrt(pi)
                        G1_elliptic_integral_factor = -(2.0*prefac*sqrt(pi))*( (2.0 - mm)*ellipe_mm - 2.0*(1.0 - mm)*ellipk_mm )/(3.0*mm)
                        #G2_elliptic_integral_factor = (2.0*prefac*sqrt(pi))*( (7.0*mm^2 + 8.0*mm - 8.0)*ellipe_mm + 4.0*(2.0 - mm)*(1.0 - mm)*ellipk_mm )/(15.0*mm^2)
                        #G3_elliptic_integral_factor = (2.0*prefac*sqrt(pi))*( 8.0*(mm^2 - mm + 1.0)*ellipe_mm - 4.0*(2.0 - mm)*(1.0 - mm)*ellipk_mm )/(15.0*mm^2)
                        H_elliptic_integral_factor = 2.0*ellipk_mm*sqrt(pi)/prefac
                        H1_elliptic_integral_factor = -(2.0*sqrt(pi)/prefac)*( (mm-2.0)*(ellipk_mm/mm) + (2.0*ellipe_mm/mm) )
                        H2_elliptic_integral_factor = (2.0*sqrt(pi)/prefac)*( (3.0*mm^2 - 8.0*mm + 8.0)*(ellipk_mm/(3.0*mm^2)) + (4.0*mm - 8.0)*ellipe_mm/(3.0*mm^2) )
                        lagrange_poly_vpa = lagrange_poly(igrid_vpa_lpoly_data,x_kvpa)
                        lagrange_poly_vperp = lagrange_poly(igrid_vperp_lpoly_data,x_kvperp)

                        (G0_weights[ivpap,ivperpp] +=
                            lagrange_poly_vpa*lagrange_poly_vperp*
                            G_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))

                        (G1_weights[ivpap,ivperpp] +=
                            lagrange_poly_vpa*lagrange_poly_vperp*
                            G1_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))

                        #(G2_weights[ivpap,ivperpp] +=
                        #    lagrange_poly_vpa*lagrange_poly_vperp*
                        #    G2_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))

                        #(G3_weights[ivpap,ivperpp] +=
                        #    lagrange_poly_vpa*lagrange_poly_vperp*
                        #    G3_elliptic_integral_factor*w_kvperp*w_kvpa*2.0/sqrt(pi))

                        (H0_weights[ivpap,ivperpp] +=
                            lagrange_poly_vpa*lagrange_poly_vperp*
                            H_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))

                        (H1_weights[ivpap,ivperpp] +=
                            lagrange_poly_vpa*lagrange_poly_vperp*
                            H1_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))

                        (H2_weights[ivpap,ivperpp] +=
                            lagrange_poly_vpa*lagrange_poly_vperp*
                            (H1_elliptic_integral_factor*vperp_val - H2_elliptic_integral_factor*x_kvperp)*
                            x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                        (H3_weights[ivpap,ivperpp] +=
                            lagrange_poly_vpa*lagrange_poly_vperp*
                            H_elliptic_integral_factor*(vpa_val - x_kvpa)*
                            x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))

                        #(n_weights[ivpap,ivperpp] +=
                        #    lagrange_poly_vpa*lagrange_poly_vperp*
                        #    x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                    end
                end
            end
        end
        return nothing
    end
end

"""
Function for computing the quadratures and carrying out the loop over the
primed `vpa` coordinate in doing the numerical integration. Splits the integrand
into three pieces -- two which use Gauss-Legendre quadrature assuming no divergences
in the integrand, and one which assumes a logarithmic divergence and uses a
Gauss-Laguerre quadrature with an (exponential) change of variables to mitigate this divergence.
"""
function loop_over_vpa_elements!(G0_weights::Twgts,G1_weights::Twgts,H0_weights::Twgts,
                            H1_weights::Twgts,H2_weights::Twgts,H3_weights::Twgts,
                            vpa::FiniteElementCoordinate,ielement_vpa_low::Int64,ielement_vpa_hi::Int64, # info about primed vperp grids
                            vperp::FiniteElementCoordinate,ielement_vperpp::Int64, # info about primed vperp grids
                            x_vpa::Tx, w_vpa::Tx, x_vperp::Tx, w_vperp::Tx, # arrays to store points and weights for primed (source) grids
                            x_legendre::Tx,w_legendre::Tx,x_laguerre::Tx,w_laguerre::Tx,
                            igrid_vpa::Int64, igrid_vperp::Int64,
                            vpa_val::Float64, vperp_val::Float64) where {Twgts <: AbstractArray{Float64,2}, Tx <: AbstractArray{Float64,1}}
    @inbounds begin
        vperp_nodes = get_nodes(vperp,ielement_vperpp)
        vperp_max = vperp_nodes[end]
        vperp_min = vperp_nodes[1]*nel_low(ielement_vperpp,vperp.nelement)
        nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
        for ielement_vpap in 1:ielement_vpa_low-1
            # do integration over part of the domain with no divergences
            vpa_nodes = get_nodes(vpa,ielement_vpap)
            vpa_min, vpa_max = vpa_nodes[1], vpa_nodes[end]
            nquad_vpa = get_scaled_x_w_no_divergences!(x_vpa, w_vpa, x_legendre, w_legendre, vpa_min, vpa_max)
            local_element_integration!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                        nquad_vpa,ielement_vpap,vpa,
                        nquad_vperp,ielement_vperpp,vperp,
                        x_vpa, w_vpa, x_vperp, w_vperp,
                        vpa_val, vperp_val)
        end
        nquad_vperp = get_scaled_x_w_with_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, x_laguerre, w_laguerre, vperp_min, vperp_max, vperp_nodes, igrid_vperp, vperp_val)
        for ielement_vpap in ielement_vpa_low:ielement_vpa_hi
        #for ielement_vpap in 1:vpa.nelement
            # use general grid function that checks divergences
            vpa_nodes = get_nodes(vpa,ielement_vpap)
            vpa_min, vpa_max = vpa_nodes[1], vpa_nodes[end]
            #nquad_vpa = get_scaled_x_w_no_divergences!(x_vpa, w_vpa, x_legendre, w_legendre, vpa_min, vpa_max)
            nquad_vpa = get_scaled_x_w_with_divergences!(x_vpa, w_vpa, x_legendre, w_legendre, x_laguerre, w_laguerre, vpa_min, vpa_max, vpa_nodes, igrid_vpa, vpa_val)
            local_element_integration!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                        nquad_vpa,ielement_vpap,vpa,
                        nquad_vperp,ielement_vperpp,vperp,
                        x_vpa, w_vpa, x_vperp, w_vperp,
                        vpa_val, vperp_val)
        end
        nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
        for ielement_vpap in ielement_vpa_hi+1:vpa.nelement
            # do integration over part of the domain with no divergences
            vpa_nodes = get_nodes(vpa,ielement_vpap)
            vpa_min, vpa_max = vpa_nodes[1], vpa_nodes[end]
            nquad_vpa = get_scaled_x_w_no_divergences!(x_vpa, w_vpa, x_legendre, w_legendre, vpa_min, vpa_max)
            local_element_integration!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                        nquad_vpa,ielement_vpap,vpa,
                        nquad_vperp,ielement_vperpp,vperp,
                        x_vpa, w_vpa, x_vperp, w_vperp,
                        vpa_val, vperp_val)

        end
        return nothing
    end
end

"""
Function for computing the quadratures and carrying out the loop over the
primed `vpa` coordinate in doing the numerical integration.
Uses a Gauss-Legendre quadrature assuming no divergences in the integrand.
"""
function loop_over_vpa_elements_no_divergences!(G0_weights::Twgts,G1_weights::Twgts,H0_weights::Twgts,
                            H1_weights::Twgts,H2_weights::Twgts,H3_weights::Twgts,
                            # info about primed vperp grids
                            vpa::FiniteElementCoordinate,ielement_vpa_low::Int64,ielement_vpa_hi::Int64,
                            # info about primed vperp grids
                            nquad_vperp::Int64,ielement_vperpp::Int64,vperp_nodes::Tx,vperp::FiniteElementCoordinate,
                            # arrays to store points and weights for primed (source) grids
                            x_vpa::Tx, w_vpa::Tx, x_vperp::Tx, w_vperp::Tx, x_legendre::Tx, w_legendre::Tx,
                            # info about unprimed grids
                            vpa_val::Float64, vperp_val::Float64) where {Twgts <: AbstractArray{Float64,2}, Tx <: AbstractArray{Float64,1}}
    @inbounds begin
        for ielement_vpap in 1:vpa.nelement
            # do integration over part of the domain with no divergences
            vpa_nodes = get_nodes(vpa,ielement_vpap)
            vpa_min, vpa_max = vpa_nodes[1], vpa_nodes[end]
            nquad_vpa = get_scaled_x_w_no_divergences!(x_vpa, w_vpa, x_legendre, w_legendre, vpa_min, vpa_max)
            local_element_integration!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                        nquad_vpa,ielement_vpap,vpa,
                        nquad_vperp,ielement_vperpp,vperp,
                        x_vpa, w_vpa, x_vperp, w_vperp,
                        vpa_val, vperp_val)

        end
        return nothing
    end
end

"""
Function for computing the quadratures and carrying out the loop over the
primed `vperp` coordinate in doing the numerical integration. Splits the integrand
into three pieces -- two which use Gauss-Legendre quadrature assuming no divergences
in the integrand, and one which assumes a logarithmic divergence and uses a
Gauss-Laguerre quadrature with an (exponential) change of variables to mitigate this divergence.
This function calls `loop_over_vpa_elements_no_divergences!()` and `loop_over_vpa_elements!()`
to carry out the primed `vpa` loop within the primed `vperp` loop.
"""
function loop_over_vperp_vpa_elements!(G0_weights::Twgts,G1_weights::Twgts,
                H0_weights::Twgts,H1_weights::Twgts,H2_weights::Twgts,H3_weights::Twgts,
                vpa::FiniteElementCoordinate,ielement_vpa_low::Int64,ielement_vpa_hi::Int64, # info about primed vpa grids
                vperp::FiniteElementCoordinate,ielement_vperp_low::Int64,ielement_vperp_hi::Int64, # info about primed vperp grids
                x_vpa::Tx, w_vpa::Tx, x_vperp::Tx, w_vperp::Tx, # arrays to store points and weights for primed (source) grids
                x_legendre::Tx,w_legendre::Tx,x_laguerre::Tx,w_laguerre::Tx,
                igrid_vpa::Int64, igrid_vperp::Int64, vpa_val::Float64, vperp_val::Float64) where {Twgts <: AbstractArray{Float64,2}, Tx <: AbstractArray{Float64,1}}
    @inbounds begin
        for ielement_vperpp in 1:ielement_vperp_low-1

            vperp_nodes = get_nodes(vperp,ielement_vperpp)
            vperp_max = vperp_nodes[end]
            vperp_min = vperp_nodes[1]*nel_low(ielement_vperpp,vperp.nelement)
            nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
            loop_over_vpa_elements_no_divergences!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                    vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                    nquad_vperp,ielement_vperpp,vperp_nodes,vperp, # info about primed vperp grids
                    x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                    x_legendre,w_legendre,
                    vpa_val, vperp_val)
        end
        for ielement_vperpp in ielement_vperp_low:ielement_vperp_hi

            #vperp_nodes = get_nodes(vperp,ielement_vperpp)
            #vperp_max = vperp_nodes[end]
            #vperp_min = vperp_nodes[1]*nel_low(ielement_vperpp,vperp.nelement)
            #nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
            #nquad_vperp = get_scaled_x_w_with_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, x_laguerre, w_laguerre, vperp_min, vperp_max, vperp_nodes, igrid_vperp, vperp_val)
            loop_over_vpa_elements!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                    vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                    vperp,ielement_vperpp, # info about primed vperp grids
                    x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                    x_legendre,w_legendre,x_laguerre,w_laguerre,
                    igrid_vpa, igrid_vperp, vpa_val, vperp_val)
        end
        for ielement_vperpp in ielement_vperp_hi+1:vperp.nelement

            vperp_nodes = get_nodes(vperp,ielement_vperpp)
            vperp_max = vperp_nodes[end]
            vperp_min = vperp_nodes[1]*nel_low(ielement_vperpp,vperp.nelement)
            nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
            loop_over_vpa_elements_no_divergences!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                    vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                    nquad_vperp,ielement_vperpp,vperp_nodes,vperp, # info about primed vperp grids
                    x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                    x_legendre,w_legendre,
                    vpa_val, vperp_val)
        end
        return nothing
    end
end

"""
The function `loop_over_vperp_vpa_elements_no_divergences!()` was used for debugging.
By changing the source where `loop_over_vperp_vpa_elements!()` is called to
instead call this function we can verify that the Gauss-Legendre quadrature
is adequate for integrating a divergence-free integrand. This function should be
kept until we understand the problems preventing machine-precision accurary in the pure integration method of computing the
Rosenbluth potentials.
"""
function loop_over_vperp_vpa_elements_no_divergences!(G0_weights::Twgts,G1_weights::Twgts,
                H0_weights::Twgts,H1_weights::Twgts,H2_weights::Twgts,H3_weights::Twgts,
                vpa::FiniteElementCoordinate,ielement_vpa_low::Int64,ielement_vpa_hi::Int64, # info about primed vpa grids
                vperp::FiniteElementCoordinate,ielement_vperp_low::Int64,ielement_vperp_hi::Int64, # info about primed vperp grids
                x_vpa::Tx, w_vpa::Tx, x_vperp::Tx, w_vperp::Tx, # arrays to store points and weights for primed (source) grids
                x_legendre::Tx,w_legendre::Tx,
                igrid_vpa::Int64, igrid_vperp::Int64, vpa_val::Float64, vperp_val::Float64) where {Twgts <: AbstractArray{Float64,2}, Tx <: AbstractArray{Float64,1}}
    @inbounds begin
        for ielement_vperpp in 1:vperp.nelement
            vperp_nodes = get_nodes(vperp,ielement_vperpp)
            vperp_max = vperp_nodes[end]
            vperp_min = vperp_nodes[1]*nel_low(ielement_vperpp,vperp.nelement)
            nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
            loop_over_vpa_elements_no_divergences!(G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                    vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                    nquad_vperp,ielement_vperpp,vperp_nodes,vperp, # info about primed vperp grids
                    x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                    x_legendre,w_legendre,
                    vpa_val, vperp_val)
        end
        return nothing
    end
end


"""
    ic_func(ivpa::Int64,ivperp::Int64,nvpa::Int64)

Get the 'linear index' corresponding to `ivpa` and `ivperp`. Defined so that the linear
index corresponds to the underlying layout in memory of a 2d array indexed by
`[ivpa,ivperp]`, i.e. for a 2d array `f2d`:
* `size(f2d) == (vpa.n, vperp.n)`
* For a reference to `f2d` that is reshaped to a vector (a 1d array) `f1d = vec(f2d)` than
  for any `ivpa` and `ivperp` it is true that `f1d[ic_func(ivpa,ivperp)] ==
  f2d[ivpa,ivperp]`.
"""
function ic_func(ivpa::Int64,ivperp::Int64,nvpa::Int64)
    return ivpa + nvpa*(ivperp-1)
end

"""
Function to assign precomputed (exact) data to an instance
of `VpaVperpBoundaryData`. Used in testing.
"""
function assign_exact_boundary_data!(func_data::VpaVperpBoundaryData,
            func_exact::Tpdf, vpa::FiniteElementCoordinate, vperp::FiniteElementCoordinate
            ) where Tpdf <: AbstractArray{Float64,2}
    nvpa = vpa.n
    nvperp = vperp.n
    @inbounds begin
        for ivperp in 1:nvperp
            func_data.lower_boundary_vpa[ivperp] = func_exact[1,ivperp]
            func_data.upper_boundary_vpa[ivperp] = func_exact[nvpa,ivperp]
        end
        for ivpa in 1:nvpa
            func_data.upper_boundary_vperp[ivpa] = func_exact[ivpa,nvperp]
        end
    end
    return nothing
end

"""
Function to assign data to an instance of `RosenbluthPotentialBoundaryData`, in place,
without allocation. Used in testing.
"""
function calculate_rosenbluth_potential_boundary_data_exact!(rpbd::RosenbluthPotentialBoundaryData,
  H_exact::Tpdf,dHdvpa_exact::Tpdf,dHdvperp_exact::Tpdf,G_exact::Tpdf,dGdvperp_exact::Tpdf,
  d2Gdvperp2_exact::Tpdf,d2Gdvperpdvpa_exact::Tpdf,d2Gdvpa2_exact::Tpdf,
  vpa::FiniteElementCoordinate,vperp::FiniteElementCoordinate) where Tpdf <: AbstractArray{Float64,2}
    assign_exact_boundary_data!(rpbd.H_data,H_exact,vpa,vperp)
    assign_exact_boundary_data!(rpbd.dHdvpa_data,dHdvpa_exact,vpa,vperp)
    assign_exact_boundary_data!(rpbd.dHdvperp_data,dHdvperp_exact,vpa,vperp)
    assign_exact_boundary_data!(rpbd.G_data,G_exact,vpa,vperp)
    assign_exact_boundary_data!(rpbd.dGdvperp_data,dGdvperp_exact,vpa,vperp)
    assign_exact_boundary_data!(rpbd.d2Gdvperp2_data,d2Gdvperp2_exact,vpa,vperp)
    assign_exact_boundary_data!(rpbd.d2Gdvperpdvpa_data,d2Gdvperpdvpa_exact,vpa,vperp)
    assign_exact_boundary_data!(rpbd.d2Gdvpa2_data,d2Gdvpa2_exact,vpa,vperp)
    return nothing
end

"""
Function to carry out the direct integration of a formal definition of one
of the Rosenbluth potentials, on the boundaries of the `(vpa,vperp)` domain,
using the precomputed integration weights with dimension 4.
The result is stored in an instance of `VpaVperpBoundaryData`.
Used in testing.
"""
function calculate_boundary_data!(func_data::VpaVperpBoundaryData,
            weight::Array{Float64,4}, func_input::Tpdf,
            vpa::FiniteElementCoordinate, vperp::FiniteElementCoordinate
            ) where Tpdf <: AbstractArray{Float64,2}
    nvpa = vpa.n
    nvperp = vperp.n
    @inbounds for ivperp in 1:vperp.n
        func_data.lower_boundary_vpa[ivperp] = 0.0
        func_data.upper_boundary_vpa[ivperp] = 0.0
        for ivperpp in 1:nvperp
            for ivpap in 1:nvpa
                func_data.lower_boundary_vpa[ivperp] += weight[ivpap,ivperpp,1,ivperp]*func_input[ivpap,ivperpp]
                func_data.upper_boundary_vpa[ivperp] += weight[ivpap,ivperpp,nvpa,ivperp]*func_input[ivpap,ivperpp]
            end
        end
    end
    #for ivpa in 1:nvpa
    @inbounds for ivpa in 1:vpa.n
        func_data.upper_boundary_vperp[ivpa] = 0.0
        for ivperpp in 1:nvperp
            for ivpap in 1:nvpa
                func_data.upper_boundary_vperp[ivpa] += weight[ivpap,ivperpp,ivpa,nvperp]*func_input[ivpap,ivperpp]
            end
        end
    end
    return nothing
end

"""
Function to carry out the direct integration of a formal definition of one
of the Rosenbluth potentials, on the boundaries of the `(vpa,vperp)` domain,
using the precomputed integration weights with dimension 3.
The result is stored in an instance of `VpaVperpBoundaryData`.
"""
function calculate_boundary_data!(func_data::VpaVperpBoundaryData,
            weight::BoundaryIntegrationWeights, func_input::Tpdf,
            vpa::FiniteElementCoordinate, vperp::FiniteElementCoordinate
            ) where Tpdf <: AbstractArray{Float64,2}
    nvpa = vpa.n
    nvperp = vperp.n
    @inbounds for ivperp in 1:vperp.n
        func_data.lower_boundary_vpa[ivperp] = 0.0
        func_data.upper_boundary_vpa[ivperp] = 0.0
        for ivperpp in 1:nvperp
            for ivpap in 1:nvpa
                func_data.lower_boundary_vpa[ivperp] += weight.lower_vpa_boundary[ivpap,ivperpp,ivperp]*func_input[ivpap,ivperpp]
                func_data.upper_boundary_vpa[ivperp] += weight.upper_vpa_boundary[ivpap,ivperpp,ivperp]*func_input[ivpap,ivperpp]
            end
        end
    end
    #for ivpa in 1:nvpa
    @inbounds for ivpa in 1:vpa.n
        func_data.upper_boundary_vperp[ivpa] = 0.0
        for ivperpp in 1:nvperp
            for ivpap in 1:nvpa
                func_data.upper_boundary_vperp[ivpa] += weight.upper_vperp_boundary[ivpap,ivperpp,ivpa]*func_input[ivpap,ivperpp]
            end
        end
    end

    return nothing
end

"""
Function to call direct integration function `calculate_boundary_data!()` and
assign data to an instance of `RosenbluthPotentialBoundaryData`, in place,
without allocation.
"""
function calculate_rosenbluth_potential_boundary_data!(rpbd::RosenbluthPotentialBoundaryData,
    fkpl::Union{FokkerPlanckArraysDirectIntegration,FokkerPlanckBoundaryIntegration},
    pdf::Tpdf,vpa::FiniteElementCoordinate,vperp::FiniteElementCoordinate;
    calculate_GG=false,calculate_dGdvperp=false
    ) where Tpdf <: AbstractArray{Float64,2}
    # get derivatives of pdf
    dfdvperp = fkpl.dfdvperp
    dfdvpa = fkpl.dfdvpa
    d2fdvperpdvpa = fkpl.d2fdvperpdvpa
    #for ivpa in 1:vpa.n
    @inbounds for ivpa in 1:vpa.n
        @views first_derivative!(dfdvperp[ivpa,:], pdf[ivpa,:], vperp)
    end
    @inbounds for ivperp in 1:vperp.n
    #for ivperp in 1:vperp.n
        @views first_derivative!(dfdvpa[:,ivperp], pdf[:,ivperp], vpa)
        @views first_derivative!(d2fdvperpdvpa[:,ivperp], dfdvperp[:,ivperp], vpa)
    end
    # carry out the numerical integration
    calculate_boundary_data!(rpbd.H_data,fkpl.H0_weights,pdf,vpa,vperp)
    calculate_boundary_data!(rpbd.dHdvpa_data,fkpl.H0_weights,dfdvpa,vpa,vperp)
    calculate_boundary_data!(rpbd.dHdvperp_data,fkpl.H1_weights,dfdvperp,vpa,vperp)
    if calculate_GG
        calculate_boundary_data!(rpbd.G_data,fkpl.G0_weights,pdf,vpa,vperp)
    end
    if calculate_dGdvperp
        calculate_boundary_data!(rpbd.dGdvperp_data,fkpl.G1_weights,dfdvperp,vpa,vperp)
    end
    calculate_boundary_data!(rpbd.d2Gdvperp2_data,fkpl.H2_weights,dfdvperp,vpa,vperp)
    calculate_boundary_data!(rpbd.d2Gdvperpdvpa_data,fkpl.G1_weights,d2fdvperpdvpa,vpa,vperp)
    calculate_boundary_data!(rpbd.d2Gdvpa2_data,fkpl.H3_weights,dfdvpa,vpa,vperp)

    return nothing
end

# types for labelling Rosenbluth potentials, for type dispatch
abstract type AbstractRosenbluthPotentialLabel end
struct HLabel <: AbstractRosenbluthPotentialLabel end
struct dHdvpaLabel <: AbstractRosenbluthPotentialLabel end
struct dHdvperpLabel <: AbstractRosenbluthPotentialLabel end
struct GLabel <: AbstractRosenbluthPotentialLabel end
struct dGdvperpLabel <: AbstractRosenbluthPotentialLabel end
struct d2Gdvperp2Label <: AbstractRosenbluthPotentialLabel end
struct d2Gdvpa2Label <: AbstractRosenbluthPotentialLabel end
struct d2GdvperpdvpaLabel <: AbstractRosenbluthPotentialLabel end

function rosenbluth_potential_Maxwellian(label::HLabel,dens::Float64,upar::Float64,vth::Float64,vpa::Float64,vperp::Float64)
    return H_Maxwellian(dens,upar,vth,vpa,vperp)
end
function rosenbluth_potential_Maxwellian(label::dHdvpaLabel,dens::Float64,upar::Float64,vth::Float64,vpa::Float64,vperp::Float64)
    return dHdvpa_Maxwellian(dens,upar,vth,vpa,vperp)
end
function rosenbluth_potential_Maxwellian(label::dHdvperpLabel,dens::Float64,upar::Float64,vth::Float64,vpa::Float64,vperp::Float64)
    return dHdvperp_Maxwellian(dens,upar,vth,vpa,vperp)
end
function rosenbluth_potential_Maxwellian(label::GLabel,dens::Float64,upar::Float64,vth::Float64,vpa::Float64,vperp::Float64)
    return G_Maxwellian(dens,upar,vth,vpa,vperp)
end
function rosenbluth_potential_Maxwellian(label::dGdvperpLabel,dens::Float64,upar::Float64,vth::Float64,vpa::Float64,vperp::Float64)
    return dGdvperp_Maxwellian(dens,upar,vth,vpa,vperp)
end
function rosenbluth_potential_Maxwellian(label::d2Gdvperp2Label,dens::Float64,upar::Float64,vth::Float64,vpa::Float64,vperp::Float64)
    return d2Gdvperp2_Maxwellian(dens,upar,vth,vpa,vperp)
end
function rosenbluth_potential_Maxwellian(label::d2GdvperpdvpaLabel,dens::Float64,upar::Float64,vth::Float64,vpa::Float64,vperp::Float64)
    return d2Gdvperpdvpa_Maxwellian(dens,upar,vth,vpa,vperp)
end
function rosenbluth_potential_Maxwellian(label::d2Gdvpa2Label,dens::Float64,upar::Float64,vth::Float64,vpa::Float64,vperp::Float64)
    return d2Gdvpa2_Maxwellian(dens,upar,vth,vpa,vperp)
end

function multipole_series(label::Tlabel,vpa::Float64,vperp::Float64,expansion_data::DeltaFMultipoleMoments
    ) where Tlabel <: AbstractRosenbluthPotentialLabel
    (dens, upar, vth) = expansion_data.Maxwellian_moments
    Inm_vec = expansion_data.Inm_vec
    series =  multipole_series(label,vpa,vperp,Inm_vec)
    series += rosenbluth_potential_Maxwellian(label,dens,upar,vth,vpa,vperp)
    return series
end
function multipole_series(label::HLabel,vpa::Float64,vperp::Float64,Inm_vec::Vector{Float64})
   (I00, I10, I20, I30, I40, I50, I60, I70, I80,
   I02, I12, I22, I32, I42, I52, I62,
   I04, I14, I24, I34, I44,
   I06, I16, I26,
   I08) = Inm_vec
   # sum up terms in the multipole series
   H_series = (I80*((128*vpa^8 - 1792*vpa^6*vperp^2 + 3360*vpa^4*vperp^4 - 1120*vpa^2*vperp^6 + 35*vperp^8)/(128*(vpa^2 + vperp^2)^8))
             +I70*((vpa*(16*vpa^6 - 168*vpa^4*vperp^2 + 210*vpa^2*vperp^4 - 35*vperp^6))/(16*(vpa^2 + vperp^2)^7))
             +I62*((-7*(128*vpa^8 - 1792*vpa^6*vperp^2 + 3360*vpa^4*vperp^4 - 1120*vpa^2*vperp^6 + 35*vperp^8))/(64*(vpa^2 + vperp^2)^8))
             +I60*((16*vpa^6 - 120*vpa^4*vperp^2 + 90*vpa^2*vperp^4 - 5*vperp^6)/(16*(vpa^2 + vperp^2)^6))
             +I52*((21*vpa*(-16*vpa^6 + 168*vpa^4*vperp^2 - 210*vpa^2*vperp^4 + 35*vperp^6))/(32*(vpa^2 + vperp^2)^7))
             +I50*((8*vpa^5 - 40*vpa^3*vperp^2 + 15*vpa*vperp^4)/(8*(vpa^2 + vperp^2)^5))
             +I44*((105*(128*vpa^8 - 1792*vpa^6*vperp^2 + 3360*vpa^4*vperp^4 - 1120*vpa^2*vperp^6 + 35*vperp^8))/(512*(vpa^2 + vperp^2)^8))
             +I42*((-15*(16*vpa^6 - 120*vpa^4*vperp^2 + 90*vpa^2*vperp^4 - 5*vperp^6))/(32*(vpa^2 + vperp^2)^6))
             +I40*((8*vpa^4 - 24*vpa^2*vperp^2 + 3*vperp^4)/(8*(vpa^2 + vperp^2)^4))
             +I34*((105*vpa*(16*vpa^6 - 168*vpa^4*vperp^2 + 210*vpa^2*vperp^4 - 35*vperp^6))/(128*(vpa^2 + vperp^2)^7))
             +I32*((-5*vpa*(8*vpa^4 - 40*vpa^2*vperp^2 + 15*vperp^4))/(8*(vpa^2 + vperp^2)^5))
             +I30*((vpa*(2*vpa^2 - 3*vperp^2))/(2*(vpa^2 + vperp^2)^3))
             +I26*((-35*(128*vpa^8 - 1792*vpa^6*vperp^2 + 3360*vpa^4*vperp^4 - 1120*vpa^2*vperp^6 + 35*vperp^8))/(512*(vpa^2 + vperp^2)^8))
             +I24*((45*(16*vpa^6 - 120*vpa^4*vperp^2 + 90*vpa^2*vperp^4 - 5*vperp^6))/(128*(vpa^2 + vperp^2)^6))
             +I22*((-3*(8*vpa^4 - 24*vpa^2*vperp^2 + 3*vperp^4))/(8*(vpa^2 + vperp^2)^4))
             +I20*(-1/2*(-2*vpa^2 + vperp^2)/(vpa^2 + vperp^2)^2)
             +I16*((-35*vpa*(16*vpa^6 - 168*vpa^4*vperp^2 + 210*vpa^2*vperp^4 - 35*vperp^6))/(256*(vpa^2 + vperp^2)^7))
             +I14*((15*vpa*(8*vpa^4 - 40*vpa^2*vperp^2 + 15*vperp^4))/(64*(vpa^2 + vperp^2)^5))
             +I12*((-6*vpa^3 + 9*vpa*vperp^2)/(4*(vpa^2 + vperp^2)^3))
             +I10*(vpa/(vpa^2 + vperp^2))
             +I08*((35*(128*vpa^8 - 1792*vpa^6*vperp^2 + 3360*vpa^4*vperp^4 - 1120*vpa^2*vperp^6 + 35*vperp^8))/(16384*(vpa^2 + vperp^2)^8))
             +I06*((-5*(16*vpa^6 - 120*vpa^4*vperp^2 + 90*vpa^2*vperp^4 - 5*vperp^6))/(256*(vpa^2 + vperp^2)^6))
             +I04*((3*(8*vpa^4 - 24*vpa^2*vperp^2 + 3*vperp^4))/(64*(vpa^2 + vperp^2)^4))
             +I02*((-2*vpa^2 + vperp^2)/(4*(vpa^2 + vperp^2)^2))
             +I00*(1))
   # multiply by overall prefactor
   H_series *= ((vpa^2 + vperp^2)^(-1/2))
   return H_series
end
function multipole_series(label::dHdvpaLabel,vpa::Float64,vperp::Float64,Inm_vec::Vector{Float64})
   (I00, I10, I20, I30, I40, I50, I60, I70, I80,
   I02, I12, I22, I32, I42, I52, I62,
   I04, I14, I24, I34, I44,
   I06, I16, I26,
   I08) = Inm_vec
   # sum up terms in the multipole series
   dHdvpa_series = (I80*((9*vpa*(128*vpa^8 - 2304*vpa^6*vperp^2 + 6048*vpa^4*vperp^4 - 3360*vpa^2*vperp^6 + 315*vperp^8))/(128*(vpa^2 + vperp^2)^8))
                +I70*((128*vpa^8 - 1792*vpa^6*vperp^2 + 3360*vpa^4*vperp^4 - 1120*vpa^2*vperp^6 + 35*vperp^8)/(16*(vpa^2 + vperp^2)^7))
                +I62*((-63*(128*vpa^9 - 2304*vpa^7*vperp^2 + 6048*vpa^5*vperp^4 - 3360*vpa^3*vperp^6 + 315*vpa*vperp^8))/(64*(vpa^2 + vperp^2)^8))
                +I60*((7*vpa*(16*vpa^6 - 168*vpa^4*vperp^2 + 210*vpa^2*vperp^4 - 35*vperp^6))/(16*(vpa^2 + vperp^2)^6))
                +I52*((-21*(128*vpa^8 - 1792*vpa^6*vperp^2 + 3360*vpa^4*vperp^4 - 1120*vpa^2*vperp^6 + 35*vperp^8))/(32*(vpa^2 + vperp^2)^7))
                +I50*((3*(16*vpa^6 - 120*vpa^4*vperp^2 + 90*vpa^2*vperp^4 - 5*vperp^6))/(8*(vpa^2 + vperp^2)^5))
                +I44*((945*vpa*(128*vpa^8 - 2304*vpa^6*vperp^2 + 6048*vpa^4*vperp^4 - 3360*vpa^2*vperp^6 + 315*vperp^8))/(512*(vpa^2 + vperp^2)^8))
                +I42*((-105*vpa*(16*vpa^6 - 168*vpa^4*vperp^2 + 210*vpa^2*vperp^4 - 35*vperp^6))/(32*(vpa^2 + vperp^2)^6))
                +I40*((5*vpa*(8*vpa^4 - 40*vpa^2*vperp^2 + 15*vperp^4))/(8*(vpa^2 + vperp^2)^4))
                +I34*((105*(128*vpa^8 - 1792*vpa^6*vperp^2 + 3360*vpa^4*vperp^4 - 1120*vpa^2*vperp^6 + 35*vperp^8))/(128*(vpa^2 + vperp^2)^7))
                +I32*((-15*(16*vpa^6 - 120*vpa^4*vperp^2 + 90*vpa^2*vperp^4 - 5*vperp^6))/(8*(vpa^2 + vperp^2)^5))
                +I30*((8*vpa^4 - 24*vpa^2*vperp^2 + 3*vperp^4)/(2*(vpa^2 + vperp^2)^3))
                +I26*((-315*vpa*(128*vpa^8 - 2304*vpa^6*vperp^2 + 6048*vpa^4*vperp^4 - 3360*vpa^2*vperp^6 + 315*vperp^8))/(512*(vpa^2 + vperp^2)^8))
                +I24*((315*vpa*(16*vpa^6 - 168*vpa^4*vperp^2 + 210*vpa^2*vperp^4 - 35*vperp^6))/(128*(vpa^2 + vperp^2)^6))
                +I22*((-15*vpa*(8*vpa^4 - 40*vpa^2*vperp^2 + 15*vperp^4))/(8*(vpa^2 + vperp^2)^4))
                +I20*((3*vpa*(2*vpa^2 - 3*vperp^2))/(2*(vpa^2 + vperp^2)^2))
                +I16*((-35*(128*vpa^8 - 1792*vpa^6*vperp^2 + 3360*vpa^4*vperp^4 - 1120*vpa^2*vperp^6 + 35*vperp^8))/(256*(vpa^2 + vperp^2)^7))
                +I14*((45*(16*vpa^6 - 120*vpa^4*vperp^2 + 90*vpa^2*vperp^4 - 5*vperp^6))/(64*(vpa^2 + vperp^2)^5))
                +I12*((-3*(8*vpa^4 - 24*vpa^2*vperp^2 + 3*vperp^4))/(4*(vpa^2 + vperp^2)^3))
                +I10*(-1 + (3*vpa^2)/(vpa^2 + vperp^2))
                +I08*((315*vpa*(128*vpa^8 - 2304*vpa^6*vperp^2 + 6048*vpa^4*vperp^4 - 3360*vpa^2*vperp^6 + 315*vperp^8))/(16384*(vpa^2 + vperp^2)^8))
                +I06*((-35*vpa*(16*vpa^6 - 168*vpa^4*vperp^2 + 210*vpa^2*vperp^4 - 35*vperp^6))/(256*(vpa^2 + vperp^2)^6))
                +I04*((15*vpa*(8*vpa^4 - 40*vpa^2*vperp^2 + 15*vperp^4))/(64*(vpa^2 + vperp^2)^4))
                +I02*((-6*vpa^3 + 9*vpa*vperp^2)/(4*(vpa^2 + vperp^2)^2))
                +I00*(vpa))
   # multiply by overall prefactor
   dHdvpa_series *= -((vpa^2 + vperp^2)^(-3/2))
   return dHdvpa_series
end
function multipole_series(label::dHdvperpLabel,vpa::Float64,vperp::Float64,Inm_vec::Vector{Float64})
   (I00, I10, I20, I30, I40, I50, I60, I70, I80,
   I02, I12, I22, I32, I42, I52, I62,
   I04, I14, I24, I34, I44,
   I06, I16, I26,
   I08) = Inm_vec
   # sum up terms in the multipole series
   dHdvperp_series = (I80*((45*vperp*(128*vpa^8 - 896*vpa^6*vperp^2 + 1120*vpa^4*vperp^4 - 280*vpa^2*vperp^6 + 7*vperp^8))/(128*(vpa^2 + vperp^2)^8))
                +I70*((9*vpa*vperp*(64*vpa^6 - 336*vpa^4*vperp^2 + 280*vpa^2*vperp^4 - 35*vperp^6))/(16*(vpa^2 + vperp^2)^7))
                +I62*((-315*(128*vpa^8*vperp - 896*vpa^6*vperp^3 + 1120*vpa^4*vperp^5 - 280*vpa^2*vperp^7 + 7*vperp^9))/(64*(vpa^2 + vperp^2)^8))
                +I60*((7*vperp*(64*vpa^6 - 240*vpa^4*vperp^2 + 120*vpa^2*vperp^4 - 5*vperp^6))/(16*(vpa^2 + vperp^2)^6))
                +I52*((-189*vpa*vperp*(64*vpa^6 - 336*vpa^4*vperp^2 + 280*vpa^2*vperp^4 - 35*vperp^6))/(32*(vpa^2 + vperp^2)^7))
                +I50*((21*vpa*vperp*(8*vpa^4 - 20*vpa^2*vperp^2 + 5*vperp^4))/(8*(vpa^2 + vperp^2)^5))
                +I44*((4725*vperp*(128*vpa^8 - 896*vpa^6*vperp^2 + 1120*vpa^4*vperp^4 - 280*vpa^2*vperp^6 + 7*vperp^8))/(512*(vpa^2 + vperp^2)^8))
                +I42*((105*vperp*(-64*vpa^6 + 240*vpa^4*vperp^2 - 120*vpa^2*vperp^4 + 5*vperp^6))/(32*(vpa^2 + vperp^2)^6))
                +I40*((15*vperp*(8*vpa^4 - 12*vpa^2*vperp^2 + vperp^4))/(8*(vpa^2 + vperp^2)^4))
                +I34*((945*vpa*vperp*(64*vpa^6 - 336*vpa^4*vperp^2 + 280*vpa^2*vperp^4 - 35*vperp^6))/(128*(vpa^2 + vperp^2)^7))
                +I32*((-105*vpa*vperp*(8*vpa^4 - 20*vpa^2*vperp^2 + 5*vperp^4))/(8*(vpa^2 + vperp^2)^5))
                +I30*((5*vpa*vperp*(4*vpa^2 - 3*vperp^2))/(2*(vpa^2 + vperp^2)^3))
                +I26*((-1575*vperp*(128*vpa^8 - 896*vpa^6*vperp^2 + 1120*vpa^4*vperp^4 - 280*vpa^2*vperp^6 + 7*vperp^8))/(512*(vpa^2 + vperp^2)^8))
                +I24*((315*vperp*(64*vpa^6 - 240*vpa^4*vperp^2 + 120*vpa^2*vperp^4 - 5*vperp^6))/(128*(vpa^2 + vperp^2)^6))
                +I22*((-45*vperp*(8*vpa^4 - 12*vpa^2*vperp^2 + vperp^4))/(8*(vpa^2 + vperp^2)^4))
                +I20*((-3*vperp*(-4*vpa^2 + vperp^2))/(2*(vpa^2 + vperp^2)^2))
                +I16*((-315*vpa*vperp*(64*vpa^6 - 336*vpa^4*vperp^2 + 280*vpa^2*vperp^4 - 35*vperp^6))/(256*(vpa^2 + vperp^2)^7))
                +I14*((315*vpa*vperp*(8*vpa^4 - 20*vpa^2*vperp^2 + 5*vperp^4))/(64*(vpa^2 + vperp^2)^5))
                +I12*((-15*vpa*vperp*(4*vpa^2 - 3*vperp^2))/(4*(vpa^2 + vperp^2)^3))
                +I10*((3*vpa*vperp)/(vpa^2 + vperp^2))
                +I08*((1575*(128*vpa^8*vperp - 896*vpa^6*vperp^3 + 1120*vpa^4*vperp^5 - 280*vpa^2*vperp^7 + 7*vperp^9))/(16384*(vpa^2 + vperp^2)^8))
                +I06*((-35*(64*vpa^6*vperp - 240*vpa^4*vperp^3 + 120*vpa^2*vperp^5 - 5*vperp^7))/(256*(vpa^2 + vperp^2)^6))
                +I04*((45*(8*vpa^4*vperp - 12*vpa^2*vperp^3 + vperp^5))/(64*(vpa^2 + vperp^2)^4))
                +I02*((3*vperp*(-4*vpa^2 + vperp^2))/(4*(vpa^2 + vperp^2)^2))
                +I00*(vperp))
   # multiply by overall prefactor
   dHdvperp_series *= -((vpa^2 + vperp^2)^(-3/2))
   return dHdvperp_series
end
function multipole_series(label::GLabel,vpa::Float64,vperp::Float64,Inm_vec::Vector{Float64})
   (I00, I10, I20, I30, I40, I50, I60, I70, I80,
   I02, I12, I22, I32, I42, I52, I62,
   I04, I14, I24, I34, I44,
   I06, I16, I26,
   I08) = Inm_vec
   # sum up terms in the multipole series
   G_series = (I80*((64*vpa^6*vperp^2 - 240*vpa^4*vperp^4 + 120*vpa^2*vperp^6 - 5*vperp^8)/(128*(vpa^2 + vperp^2)^8))
             +I70*((vpa*vperp^2*(8*vpa^4 - 20*vpa^2*vperp^2 + 5*vperp^4))/(16*(vpa^2 + vperp^2)^7))
             +I62*((32*vpa^8 - 656*vpa^6*vperp^2 + 1620*vpa^4*vperp^4 - 670*vpa^2*vperp^6 + 25*vperp^8)/(64*(vpa^2 + vperp^2)^8))
             +I60*((vperp^2*(8*vpa^4 - 12*vpa^2*vperp^2 + vperp^4))/(16*(vpa^2 + vperp^2)^6))
             +I52*((vpa*(16*vpa^6 - 232*vpa^4*vperp^2 + 370*vpa^2*vperp^4 - 75*vperp^6))/(32*(vpa^2 + vperp^2)^7))
             +I50*((vpa*vperp^2*(4*vpa^2 - 3*vperp^2))/(8*(vpa^2 + vperp^2)^5))
             +I44*((-15*(64*vpa^8 - 864*vpa^6*vperp^2 + 1560*vpa^4*vperp^4 - 500*vpa^2*vperp^6 + 15*vperp^8))/(512*(vpa^2 + vperp^2)^8))
             +I42*((16*vpa^6 - 152*vpa^4*vperp^2 + 138*vpa^2*vperp^4 - 9*vperp^6)/(32*(vpa^2 + vperp^2)^6))
             +I40*(-1/8*(vperp^2*(-4*vpa^2 + vperp^2))/(vpa^2 + vperp^2)^4)
             +I34*((5*vpa*(-32*vpa^6 + 296*vpa^4*vperp^2 - 320*vpa^2*vperp^4 + 45*vperp^6))/(128*(vpa^2 + vperp^2)^7))
             +I32*((vpa*(4*vpa^4 - 22*vpa^2*vperp^2 + 9*vperp^4))/(8*(vpa^2 + vperp^2)^5))
             +I30*((vpa*vperp^2)/(2*(vpa^2 + vperp^2)^3))
             +I26*((5*(96*vpa^8 - 1072*vpa^6*vperp^2 + 1500*vpa^4*vperp^4 - 330*vpa^2*vperp^6 + 5*vperp^8))/(512*(vpa^2 + vperp^2)^8))
             +I24*((3*(-32*vpa^6 + 184*vpa^4*vperp^2 - 96*vpa^2*vperp^4 + 3*vperp^6))/(128*(vpa^2 + vperp^2)^6))
             +I22*((4*vpa^4 - 10*vpa^2*vperp^2 + vperp^4)/(8*(vpa^2 + vperp^2)^4))
             +I20*(vperp^2/(2*(vpa^2 + vperp^2)^2))
             +I16*((5*vpa*(16*vpa^6 - 120*vpa^4*vperp^2 + 90*vpa^2*vperp^4 - 5*vperp^6))/(256*(vpa^2 + vperp^2)^7))
             +I14*((-3*vpa*(8*vpa^4 - 24*vpa^2*vperp^2 + 3*vperp^4))/(64*(vpa^2 + vperp^2)^5))
             +I12*((vpa*(2*vpa^2 - vperp^2))/(4*(vpa^2 + vperp^2)^3))
             +I10*(-(vpa/(vpa^2 + vperp^2)))
             +I08*((5*(-128*vpa^8 + 1280*vpa^6*vperp^2 - 1440*vpa^4*vperp^4 + 160*vpa^2*vperp^6 + 5*vperp^8))/(16384*(vpa^2 + vperp^2)^8))
             +I06*((16*vpa^6 - 72*vpa^4*vperp^2 + 18*vpa^2*vperp^4 + vperp^6)/(256*(vpa^2 + vperp^2)^6))
             +I04*((-8*vpa^4 + 8*vpa^2*vperp^2 + vperp^4)/(64*(vpa^2 + vperp^2)^4))
             +I02*((2*vpa^2 + vperp^2)/(4*(vpa^2 + vperp^2)^2))
             +I00*(1))
   # multiply by overall prefactor
   G_series *= ((vpa^2 + vperp^2)^(1/2))
   return G_series
end
function multipole_series(label::dGdvperpLabel,vpa::Float64,vperp::Float64,Inm_vec::Vector{Float64})
   (I00, I10, I20, I30, I40, I50, I60, I70, I80,
   I02, I12, I22, I32, I42, I52, I62,
   I04, I14, I24, I34, I44,
   I06, I16, I26,
   I08) = Inm_vec
   # sum up terms in the multipole series
   dGdvperp_series = (I80*((vperp*(128*vpa^8 - 1792*vpa^6*vperp^2 + 3360*vpa^4*vperp^4 - 1120*vpa^2*vperp^6 + 35*vperp^8))/(128*(vpa^2 + vperp^2)^8))
                   +I70*((vpa*vperp*(16*vpa^6 - 168*vpa^4*vperp^2 + 210*vpa^2*vperp^4 - 35*vperp^6))/(16*(vpa^2 + vperp^2)^7))
                   +I62*((-7*(256*vpa^8*vperp - 2144*vpa^6*vperp^3 + 3120*vpa^4*vperp^5 - 890*vpa^2*vperp^7 + 25*vperp^9))/(64*(vpa^2 + vperp^2)^8))
                   +I60*((vperp*(16*vpa^6 - 120*vpa^4*vperp^2 + 90*vpa^2*vperp^4 - 5*vperp^6))/(16*(vpa^2 + vperp^2)^6))
                   +I52*((21*vpa*vperp*(-32*vpa^6 + 192*vpa^4*vperp^2 - 180*vpa^2*vperp^4 + 25*vperp^6))/(32*(vpa^2 + vperp^2)^7))
                   +I50*((8*vpa^5*vperp - 40*vpa^3*vperp^3 + 15*vpa*vperp^5)/(8*(vpa^2 + vperp^2)^5))
                   +I44*((315*vperp*(128*vpa^8 - 832*vpa^6*vperp^2 + 960*vpa^4*vperp^4 - 220*vpa^2*vperp^6 + 5*vperp^8))/(512*(vpa^2 + vperp^2)^8))
                   +I42*((15*vperp*(-32*vpa^6 + 128*vpa^4*vperp^2 - 68*vpa^2*vperp^4 + 3*vperp^6))/(32*(vpa^2 + vperp^2)^6))
                   +I40*((vperp*(8*vpa^4 - 24*vpa^2*vperp^2 + 3*vperp^4))/(8*(vpa^2 + vperp^2)^4))
                   +I34*((315*vpa*vperp*(16*vpa^6 - 72*vpa^4*vperp^2 + 50*vpa^2*vperp^4 - 5*vperp^6))/(128*(vpa^2 + vperp^2)^7))
                   +I32*((-5*vpa*vperp*(16*vpa^4 - 38*vpa^2*vperp^2 + 9*vperp^4))/(8*(vpa^2 + vperp^2)^5))
                   +I30*((vpa*vperp*(2*vpa^2 - 3*vperp^2))/(2*(vpa^2 + vperp^2)^3))
                   +I26*((-35*vperp*(512*vpa^8 - 2848*vpa^6*vperp^2 + 2640*vpa^4*vperp^4 - 430*vpa^2*vperp^6 + 5*vperp^8))/(512*(vpa^2 + vperp^2)^8))
                   +I24*((-45*vperp*(-48*vpa^6 + 136*vpa^4*vperp^2 - 46*vpa^2*vperp^4 + vperp^6))/(128*(vpa^2 + vperp^2)^6))
                   +I22*((-3*vperp*(16*vpa^4 - 18*vpa^2*vperp^2 + vperp^4))/(8*(vpa^2 + vperp^2)^4))
                   +I20*(-1/2*(vperp*(-2*vpa^2 + vperp^2))/(vpa^2 + vperp^2)^2)
                   +I16*((-35*vpa*vperp*(64*vpa^6 - 240*vpa^4*vperp^2 + 120*vpa^2*vperp^4 - 5*vperp^6))/(256*(vpa^2 + vperp^2)^7))
                   +I14*((45*vpa*vperp*(8*vpa^4 - 12*vpa^2*vperp^2 + vperp^4))/(64*(vpa^2 + vperp^2)^5))
                   +I12*((3*vpa*vperp*(-4*vpa^2 + vperp^2))/(4*(vpa^2 + vperp^2)^3))
                   +I10*((vpa*vperp)/(vpa^2 + vperp^2))
                   +I08*((175*(128*vpa^8*vperp - 640*vpa^6*vperp^3 + 480*vpa^4*vperp^5 - 40*vpa^2*vperp^7 - vperp^9))/(16384*(vpa^2 + vperp^2)^8))
                   +I06*((-5*(64*vpa^6*vperp - 144*vpa^4*vperp^3 + 24*vpa^2*vperp^5 + vperp^7))/(256*(vpa^2 + vperp^2)^6))
                   +I04*((3*(24*vpa^4*vperp - 12*vpa^2*vperp^3 - vperp^5))/(64*(vpa^2 + vperp^2)^4))
                   +I02*(-1/4*(vperp*(4*vpa^2 + vperp^2))/(vpa^2 + vperp^2)^2)
                   +I00*(vperp))
   # multiply by overall prefactor
   dGdvperp_series *= ((vpa^2 + vperp^2)^(-1/2))
   return dGdvperp_series
end
function multipole_series(label::d2Gdvperp2Label,vpa::Float64,vperp::Float64,Inm_vec::Vector{Float64})
   (I00, I10, I20, I30, I40, I50, I60, I70, I80,
   I02, I12, I22, I32, I42, I52, I62,
   I04, I14, I24, I34, I44,
   I06, I16, I26,
   I08) = Inm_vec
   # sum up terms in the multipole series
   d2Gdvperp2_series = (I80*((128*vpa^10 - 7424*vpa^8*vperp^2 + 41888*vpa^6*vperp^4 - 48160*vpa^4*vperp^6 + 11515*vpa^2*vperp^8 - 280*vperp^10)/(128*(vpa^2 + vperp^2)^8))
                   +I70*((16*vpa^9 - 728*vpa^7*vperp^2 + 3066*vpa^5*vperp^4 - 2345*vpa^3*vperp^6 + 280*vpa*vperp^8)/(16*(vpa^2 + vperp^2)^7))
                   +I62*((-7*(256*vpa^10 - 10528*vpa^8*vperp^2 + 45616*vpa^6*vperp^4 - 43670*vpa^4*vperp^6 + 9125*vpa^2*vperp^8 - 200*vperp^10))/(64*(vpa^2 + vperp^2)^8))
                   +I60*((16*vpa^8 - 552*vpa^6*vperp^2 + 1650*vpa^4*vperp^4 - 755*vpa^2*vperp^6 + 30*vperp^8)/(16*(vpa^2 + vperp^2)^6))
                   +I52*((-21*(32*vpa^9 - 1024*vpa^7*vperp^2 + 3204*vpa^5*vperp^4 - 1975*vpa^3*vperp^6 + 200*vpa*vperp^8))/(32*(vpa^2 + vperp^2)^7))
                   +I50*((8*vpa^7 - 200*vpa^5*vperp^2 + 395*vpa^3*vperp^4 - 90*vpa*vperp^6)/(8*(vpa^2 + vperp^2)^5))
                   +I44*((315*(128*vpa^10 - 4544*vpa^8*vperp^2 + 16448*vpa^6*vperp^4 - 13060*vpa^4*vperp^6 + 2245*vpa^2*vperp^8 - 40*vperp^10))/(512*(vpa^2 + vperp^2)^8))
                   +I42*((-15*(32*vpa^8 - 768*vpa^6*vperp^2 + 1620*vpa^4*vperp^4 - 565*vpa^2*vperp^6 + 18*vperp^8))/(32*(vpa^2 + vperp^2)^6))
                   +I40*((8*vpa^6 - 136*vpa^4*vperp^2 + 159*vpa^2*vperp^4 - 12*vperp^6)/(8*(vpa^2 + vperp^2)^4))
                   +I34*((315*vpa*(16*vpa^8 - 440*vpa^6*vperp^2 + 1114*vpa^4*vperp^4 - 535*vpa^2*vperp^6 + 40*vperp^8))/(128*(vpa^2 + vperp^2)^7))
                   +I32*((5*vpa*(-16*vpa^6 + 274*vpa^4*vperp^2 - 349*vpa^2*vperp^4 + 54*vperp^6))/(8*(vpa^2 + vperp^2)^5))
                   +I30*((vpa*(2*vpa^4 - 21*vpa^2*vperp^2 + 12*vperp^4))/(2*(vpa^2 + vperp^2)^3))
                   +I26*((-35*(512*vpa^10 - 16736*vpa^8*vperp^2 + 53072*vpa^6*vperp^4 - 34690*vpa^4*vperp^6 + 4345*vpa^2*vperp^8 - 40*vperp^10))/(512*(vpa^2 + vperp^2)^8))
                   +I24*((135*(16*vpa^8 - 328*vpa^6*vperp^2 + 530*vpa^4*vperp^4 - 125*vpa^2*vperp^6 + 2*vperp^8))/(128*(vpa^2 + vperp^2)^6))
                   +I22*((-3*(16*vpa^6 - 182*vpa^4*vperp^2 + 113*vpa^2*vperp^4 - 4*vperp^6))/(8*(vpa^2 + vperp^2)^4))
                   +I20*((2*vpa^4 - 11*vpa^2*vperp^2 + 2*vperp^4)/(2*(vpa^2 + vperp^2)^2))
                   +I16*((-35*vpa*(64*vpa^8 - 1616*vpa^6*vperp^2 + 3480*vpa^4*vperp^4 - 1235*vpa^2*vperp^6 + 40*vperp^8))/(256*(vpa^2 + vperp^2)^7))
                   +I14*((45*vpa*(8*vpa^6 - 116*vpa^4*vperp^2 + 101*vpa^2*vperp^4 - 6*vperp^6))/(64*(vpa^2 + vperp^2)^5))
                   +I12*((-3*vpa*(4*vpa^4 - 27*vpa^2*vperp^2 + 4*vperp^4))/(4*(vpa^2 + vperp^2)^3))
                   +I10*(-2*vpa + (3*vpa^3)/(vpa^2 + vperp^2))
                   +I08*((175*(128*vpa^10 - 3968*vpa^8*vperp^2 + 11360*vpa^6*vperp^4 - 6040*vpa^4*vperp^6 + 391*vpa^2*vperp^8 + 8*vperp^10))/(16384*(vpa^2 + vperp^2)^8))
                   +I06*((-5*(64*vpa^8 - 1200*vpa^6*vperp^2 + 1560*vpa^4*vperp^4 - 185*vpa^2*vperp^6 - 6*vperp^8))/(256*(vpa^2 + vperp^2)^6))
                   +I04*((3*(24*vpa^6 - 228*vpa^4*vperp^2 + 67*vpa^2*vperp^4 + 4*vperp^6))/(64*(vpa^2 + vperp^2)^4))
                   +I02*((-4*vpa^4 + 13*vpa^2*vperp^2 + 2*vperp^4)/(4*(vpa^2 + vperp^2)^2))
                   +I00*(vpa^2))
   # multiply by overall prefactor
   d2Gdvperp2_series *= ((vpa^2 + vperp^2)^(-3/2))
   return d2Gdvperp2_series
end
function multipole_series(label::d2GdvperpdvpaLabel,vpa::Float64,vperp::Float64,Inm_vec::Vector{Float64})
   (I00, I10, I20, I30, I40, I50, I60, I70, I80,
   I02, I12, I22, I32, I42, I52, I62,
   I04, I14, I24, I34, I44,
   I06, I16, I26,
   I08) = Inm_vec
   # sum up terms in the multipole series
   d2Gdvperpdvpa_series = (I80*((9*vpa*vperp*(128*vpa^8 - 2304*vpa^6*vperp^2 + 6048*vpa^4*vperp^4 - 3360*vpa^2*vperp^6 + 315*vperp^8))/(128*(vpa^2 + vperp^2)^8))
                      +I70*((vperp*(128*vpa^8 - 1792*vpa^6*vperp^2 + 3360*vpa^4*vperp^4 - 1120*vpa^2*vperp^6 + 35*vperp^8))/(16*(vpa^2 + vperp^2)^7))
                      +I62*((-63*(256*vpa^9*vperp - 2848*vpa^7*vperp^3 + 5936*vpa^5*vperp^5 - 2870*vpa^3*vperp^7 + 245*vpa*vperp^9))/(64*(vpa^2 + vperp^2)^8))
                      +I60*((7*vpa*vperp*(16*vpa^6 - 168*vpa^4*vperp^2 + 210*vpa^2*vperp^4 - 35*vperp^6))/(16*(vpa^2 + vperp^2)^6))
                      +I52*((-21*(256*vpa^8*vperp - 2144*vpa^6*vperp^3 + 3120*vpa^4*vperp^5 - 890*vpa^2*vperp^7 + 25*vperp^9))/(32*(vpa^2 + vperp^2)^7))
                      +I50*((3*vperp*(16*vpa^6 - 120*vpa^4*vperp^2 + 90*vpa^2*vperp^4 - 5*vperp^6))/(8*(vpa^2 + vperp^2)^5))
                      +I44*((945*vpa*vperp*(384*vpa^8 - 3392*vpa^6*vperp^2 + 5824*vpa^4*vperp^4 - 2380*vpa^2*vperp^6 + 175*vperp^8))/(512*(vpa^2 + vperp^2)^8))
                      +I42*((-105*vpa*vperp*(32*vpa^6 - 192*vpa^4*vperp^2 + 180*vpa^2*vperp^4 - 25*vperp^6))/(32*(vpa^2 + vperp^2)^6))
                      +I40*((5*vpa*vperp*(8*vpa^4 - 40*vpa^2*vperp^2 + 15*vperp^4))/(8*(vpa^2 + vperp^2)^4))
                      +I34*((315*vperp*(128*vpa^8 - 832*vpa^6*vperp^2 + 960*vpa^4*vperp^4 - 220*vpa^2*vperp^6 + 5*vperp^8))/(128*(vpa^2 + vperp^2)^7))
                      +I32*((15*vperp*(-32*vpa^6 + 128*vpa^4*vperp^2 - 68*vpa^2*vperp^4 + 3*vperp^6))/(8*(vpa^2 + vperp^2)^5))
                      +I30*((vperp*(8*vpa^4 - 24*vpa^2*vperp^2 + 3*vperp^4))/(2*(vpa^2 + vperp^2)^3))
                      +I26*((-315*vpa*vperp*(512*vpa^8 - 3936*vpa^6*vperp^2 + 5712*vpa^4*vperp^4 - 1890*vpa^2*vperp^6 + 105*vperp^8))/(512*(vpa^2 + vperp^2)^8))
                      +I24*((945*vpa*vperp*(16*vpa^6 - 72*vpa^4*vperp^2 + 50*vpa^2*vperp^4 - 5*vperp^6))/(128*(vpa^2 + vperp^2)^6))
                      +I22*((-15*vpa*vperp*(16*vpa^4 - 38*vpa^2*vperp^2 + 9*vperp^4))/(8*(vpa^2 + vperp^2)^4))
                      +I20*((3*vpa*vperp*(2*vpa^2 - 3*vperp^2))/(2*(vpa^2 + vperp^2)^2))
                      +I16*((-35*vperp*(512*vpa^8 - 2848*vpa^6*vperp^2 + 2640*vpa^4*vperp^4 - 430*vpa^2*vperp^6 + 5*vperp^8))/(256*(vpa^2 + vperp^2)^7))
                      +I14*((-45*vperp*(-48*vpa^6 + 136*vpa^4*vperp^2 - 46*vpa^2*vperp^4 + vperp^6))/(64*(vpa^2 + vperp^2)^5))
                      +I12*((-3*vperp*(16*vpa^4 - 18*vpa^2*vperp^2 + vperp^4))/(4*(vpa^2 + vperp^2)^3))
                      +I10*(vperp*(-1 + (3*vpa^2)/(vpa^2 + vperp^2)))
                      +I08*((1575*vpa*(128*vpa^8*vperp - 896*vpa^6*vperp^3 + 1120*vpa^4*vperp^5 - 280*vpa^2*vperp^7 + 7*vperp^9))/(16384*(vpa^2 + vperp^2)^8))
                      +I06*((-35*vpa*(64*vpa^6*vperp - 240*vpa^4*vperp^3 + 120*vpa^2*vperp^5 - 5*vperp^7))/(256*(vpa^2 + vperp^2)^6))
                      +I04*((45*vpa*(8*vpa^4*vperp - 12*vpa^2*vperp^3 + vperp^5))/(64*(vpa^2 + vperp^2)^4))
                      +I02*((3*vpa*vperp*(-4*vpa^2 + vperp^2))/(4*(vpa^2 + vperp^2)^2))
                      +I00*(vpa*vperp))
   # multiply by overall prefactor
   d2Gdvperpdvpa_series *= -((vpa^2 + vperp^2)^(-3/2))
   return d2Gdvperpdvpa_series
end
function multipole_series(label::d2Gdvpa2Label,vpa::Float64,vperp::Float64,Inm_vec::Vector{Float64})
   (I00, I10, I20, I30, I40, I50, I60, I70, I80,
   I02, I12, I22, I32, I42, I52, I62,
   I04, I14, I24, I34, I44,
   I06, I16, I26,
   I08) = Inm_vec
   # sum up terms in the multipole series
   d2Gdvpa2_series = (I80*((45*vperp^2*(128*vpa^8 - 896*vpa^6*vperp^2 + 1120*vpa^4*vperp^4 - 280*vpa^2*vperp^6 + 7*vperp^8))/(128*(vpa^2 + vperp^2)^8))
                   +I70*((9*vpa*vperp^2*(64*vpa^6 - 336*vpa^4*vperp^2 + 280*vpa^2*vperp^4 - 35*vperp^6))/(16*(vpa^2 + vperp^2)^7))
                   +I62*((7*(256*vpa^10 - 9088*vpa^8*vperp^2 + 43456*vpa^6*vperp^4 - 45920*vpa^4*vperp^6 + 10430*vpa^2*vperp^8 - 245*vperp^10))/(64*(vpa^2 + vperp^2)^8))
                   +I60*((7*vperp^2*(64*vpa^6 - 240*vpa^4*vperp^2 + 120*vpa^2*vperp^4 - 5*vperp^6))/(16*(vpa^2 + vperp^2)^6))
                   +I52*((21*vpa*(32*vpa^8 - 880*vpa^6*vperp^2 + 3108*vpa^4*vperp^4 - 2170*vpa^2*vperp^6 + 245*vperp^8))/(32*(vpa^2 + vperp^2)^7))
                   +I50*((21*vpa*vperp^2*(8*vpa^4 - 20*vpa^2*vperp^2 + 5*vperp^4))/(8*(vpa^2 + vperp^2)^5))
                   +I44*((105*(-512*vpa^10 + 12416*vpa^8*vperp^2 - 46592*vpa^6*vperp^4 + 41440*vpa^4*vperp^6 - 8260*vpa^2*vperp^8 + 175*vperp^10))/(512*(vpa^2 + vperp^2)^8))
                   +I42*((15*(32*vpa^8 - 656*vpa^6*vperp^2 + 1620*vpa^4*vperp^4 - 670*vpa^2*vperp^6 + 25*vperp^8))/(32*(vpa^2 + vperp^2)^6))
                   +I40*((15*vperp^2*(8*vpa^4 - 12*vpa^2*vperp^2 + vperp^4))/(8*(vpa^2 + vperp^2)^4))
                   +I34*((-105*vpa*(64*vpa^8 - 1184*vpa^6*vperp^2 + 3192*vpa^4*vperp^4 - 1820*vpa^2*vperp^6 + 175*vperp^8))/(128*(vpa^2 + vperp^2)^7))
                   +I32*((5*vpa*(16*vpa^6 - 232*vpa^4*vperp^2 + 370*vpa^2*vperp^4 - 75*vperp^6))/(8*(vpa^2 + vperp^2)^5))
                   +I30*((5*vpa*vperp^2*(4*vpa^2 - 3*vperp^2))/(2*(vpa^2 + vperp^2)^3))
                   +I26*((105*(256*vpa^10 - 5248*vpa^8*vperp^2 + 16576*vpa^6*vperp^4 - 12320*vpa^4*vperp^6 + 2030*vpa^2*vperp^8 - 35*vperp^10))/(512*(vpa^2 + vperp^2)^8))
                   +I24*((-45*(64*vpa^8 - 864*vpa^6*vperp^2 + 1560*vpa^4*vperp^4 - 500*vpa^2*vperp^6 + 15*vperp^8))/(128*(vpa^2 + vperp^2)^6))
                   +I22*((3*(16*vpa^6 - 152*vpa^4*vperp^2 + 138*vpa^2*vperp^4 - 9*vperp^6))/(8*(vpa^2 + vperp^2)^4))
                   +I20*((-3*vperp^2*(-4*vpa^2 + vperp^2))/(2*(vpa^2 + vperp^2)^2))
                   +I16*((105*vpa*(32*vpa^8 - 496*vpa^6*vperp^2 + 1092*vpa^4*vperp^4 - 490*vpa^2*vperp^6 + 35*vperp^8))/(256*(vpa^2 + vperp^2)^7))
                   +I14*((15*vpa*(-32*vpa^6 + 296*vpa^4*vperp^2 - 320*vpa^2*vperp^4 + 45*vperp^6))/(64*(vpa^2 + vperp^2)^5))
                   +I12*((3*vpa*(4*vpa^4 - 22*vpa^2*vperp^2 + 9*vperp^4))/(4*(vpa^2 + vperp^2)^3))
                   +I10*((3*vpa*vperp^2)/(vpa^2 + vperp^2))
                   +I08*((-35*(1024*vpa^10 - 19072*vpa^8*vperp^2 + 52864*vpa^6*vperp^4 - 32480*vpa^4*vperp^6 + 3920*vpa^2*vperp^8 - 35*vperp^10))/(16384*(vpa^2 + vperp^2)^8))
                   +I06*((5*(96*vpa^8 - 1072*vpa^6*vperp^2 + 1500*vpa^4*vperp^4 - 330*vpa^2*vperp^6 + 5*vperp^8))/(256*(vpa^2 + vperp^2)^6))
                   +I04*((-3*(32*vpa^6 - 184*vpa^4*vperp^2 + 96*vpa^2*vperp^4 - 3*vperp^6))/(64*(vpa^2 + vperp^2)^4))
                   +I02*((4*vpa^4 - 10*vpa^2*vperp^2 + vperp^4)/(4*(vpa^2 + vperp^2)^2))
                   +I00*(vperp^2))
   # multiply by overall prefactor
   d2Gdvpa2_series *= ((vpa^2 + vperp^2)^(-3/2))
   return d2Gdvpa2_series
end

"""
"""
function calculate_boundary_data_multipole!(func_data::VpaVperpBoundaryData,
            label::Tlabel,
            vpa::FiniteElementCoordinate,
            vperp::FiniteElementCoordinate,
            Inm_vec::Union{Vector{Float64},DeltaFMultipoleMoments}
            ) where Tlabel <: AbstractRosenbluthPotentialLabel
    nvpa = vpa.n
    nvperp = vperp.n
    @inbounds for ivperp in 1:vperp.n
                func_data.lower_boundary_vpa[ivperp] = multipole_series(label,vpa.grid[1],vperp.grid[ivperp],Inm_vec)
                func_data.upper_boundary_vpa[ivperp] = multipole_series(label,vpa.grid[nvpa],vperp.grid[ivperp],Inm_vec)
    end
    @inbounds for ivpa in 1:vpa.n
                func_data.upper_boundary_vperp[ivpa] = multipole_series(label,vpa.grid[ivpa],vperp.grid[nvperp],Inm_vec)
    end
    return nothing
end

function calculate_multipole_expansion_moments!(Inm_vec::Vector{Float64},pdf::Tpdf,
            vpa::FiniteElementCoordinate,vperp::FiniteElementCoordinate
            ) where Tpdf <: AbstractArray{Float64,2}
    @inbounds begin
        # get required moments of pdf for the multipole expansion
        I00 = get_nm_moment(0, 0, pdf, vpa, vperp)
        I10 = get_nm_moment(1, 0, pdf, vpa, vperp)
        I20 = get_nm_moment(2, 0, pdf, vpa, vperp)
        I30 = get_nm_moment(3, 0, pdf, vpa, vperp)
        I40 = get_nm_moment(4, 0, pdf, vpa, vperp)
        I50 = get_nm_moment(5, 0, pdf, vpa, vperp)
        I60 = get_nm_moment(6, 0, pdf, vpa, vperp)
        I70 = get_nm_moment(7, 0, pdf, vpa, vperp)
        I80 = get_nm_moment(8, 0, pdf, vpa, vperp)

        I02 = get_nm_moment(0, 2, pdf, vpa, vperp)
        I12 = get_nm_moment(1, 2, pdf, vpa, vperp)
        I22 = get_nm_moment(2, 2, pdf, vpa, vperp)
        I32 = get_nm_moment(3, 2, pdf, vpa, vperp)
        I42 = get_nm_moment(4, 2, pdf, vpa, vperp)
        I52 = get_nm_moment(5, 2, pdf, vpa, vperp)
        I62 = get_nm_moment(6, 2, pdf, vpa, vperp)

        I04 = get_nm_moment(0, 4, pdf, vpa, vperp)
        I14 = get_nm_moment(1, 4, pdf, vpa, vperp)
        I24 = get_nm_moment(2, 4, pdf, vpa, vperp)
        I34 = get_nm_moment(3, 4, pdf, vpa, vperp)
        I44 = get_nm_moment(4, 4, pdf, vpa, vperp)

        I06 = get_nm_moment(0, 6, pdf, vpa, vperp)
        I16 = get_nm_moment(1, 6, pdf, vpa, vperp)
        I26 = get_nm_moment(2, 6, pdf, vpa, vperp)

        I08 = get_nm_moment(0, 8, pdf, vpa, vperp)
        # group into vector to pass around
        Inm_vec .= [I00, I10, I20, I30, I40, I50, I60, I70, I80,
                    I02, I12, I22, I32, I42, I52, I62,
                    I04, I14, I24, I34, I44,
                    I06, I16, I26,
                    I08]
    end
    return nothing
end
function calculate_multipole_expansion_moments!(expansion_data::DeltaFMultipoleMoments,
            pdf::Tpdf,dummy_vpavperp::Array{Float64,2},
            vpa::FiniteElementCoordinate,vperp::FiniteElementCoordinate
            ) where Tpdf <: AbstractArray{Float64,2}
    Inm_vec = expansion_data.Inm_vec
    Maxwellian_moments = expansion_data.Maxwellian_moments
    dens = get_density(pdf, vpa, vperp)
    upar = get_upar(pdf, vpa, vperp, dens)
    mass = 1.0 # since we only divide by, then multiply by mass in the next two lines
    pressure = get_pressure(pdf, vpa, vperp, upar, mass)
    vth = sqrt(2.0*pressure/(dens*mass))
    ppar = get_ppar(pdf, vpa, vperp, upar, mass)
    pperp = get_pperp(pressure, ppar)
    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                dummy_vpavperp[ivpa,ivperp] = pdf[ivpa,ivperp] - F_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
            end
        end
    end
    # store Maxwellian moments for use elsewhere
    @. Maxwellian_moments = [dens, upar, vth]
    # get Inm from delta f
    calculate_multipole_expansion_moments!(Inm_vec,dummy_vpavperp,vpa,vperp)
    return nothing
end

"""
Function to use the multipole expansion of the Rosenbluth potentials to calculate and
assign boundary data to an instance of `RosenbluthPotentialBoundaryData`, in place,
without allocation.
"""
function calculate_rosenbluth_potential_boundary_data_multipole!(rpbd::RosenbluthPotentialBoundaryData,
    expansion_data::Union{Vector{Float64},DeltaFMultipoleMoments},vpa::FiniteElementCoordinate,vperp::FiniteElementCoordinate;
    calculate_GG=false,calculate_dGdvperp=false)
    # evaluate the multipole formulae
    calculate_boundary_data_multipole!(rpbd.H_data,HLabel(),vpa,vperp,expansion_data)
    calculate_boundary_data_multipole!(rpbd.dHdvpa_data,dHdvpaLabel(),vpa,vperp,expansion_data)
    calculate_boundary_data_multipole!(rpbd.dHdvperp_data,dHdvperpLabel(),vpa,vperp,expansion_data)
    if calculate_GG
        calculate_boundary_data_multipole!(rpbd.G_data,GLabel(),vpa,vperp,expansion_data)
    end
    if calculate_dGdvperp
        calculate_boundary_data_multipole!(rpbd.dGdvperp_data,dGdvperpLabel(),vpa,vperp,expansion_data)
    end
    calculate_boundary_data_multipole!(rpbd.d2Gdvperp2_data,d2Gdvperp2Label(),vpa,vperp,expansion_data)
    calculate_boundary_data_multipole!(rpbd.d2Gdvperpdvpa_data,d2GdvperpdvpaLabel(),vpa,vperp,expansion_data)
    calculate_boundary_data_multipole!(rpbd.d2Gdvpa2_data,d2Gdvpa2Label(),vpa,vperp,expansion_data)
    return nothing
end
function calculate_rosenbluth_potential_boundary_data_multipole!(rpbd::RosenbluthPotentialBoundaryData,
    Inm_vec::Vector{Float64},pdf::Tpdf,
    vpa::FiniteElementCoordinate,vperp::FiniteElementCoordinate;
    calculate_GG=false,calculate_dGdvperp=false) where Tpdf <: AbstractArray{Float64,2}
    # get required moments of pdf
    calculate_multipole_expansion_moments!(Inm_vec,pdf,vpa,vperp)
    # evaluate the multipole formulae
    calculate_rosenbluth_potential_boundary_data_multipole!(rpbd,Inm_vec,vpa,vperp,
      calculate_GG=calculate_GG,calculate_dGdvperp=calculate_dGdvperp)
    return nothing
end

"""
Function to use the multipole expansion of the Rosenbluth potentials to calculate and
assign boundary data to an instance of `RosenbluthPotentialBoundaryData`, in place,
without allocation. Use the exact results for the part of F that can be described with
a Maxwellian, and the multipole expansion for the remainder.
"""
function calculate_rosenbluth_potential_boundary_data_delta_f_multipole!(rpbd::RosenbluthPotentialBoundaryData,
    expansion_data::DeltaFMultipoleMoments,
    pdf::Tpdf,dummy_vpavperp::Array{Float64,2},
    vpa::FiniteElementCoordinate,vperp::FiniteElementCoordinate;
    calculate_GG=false,calculate_dGdvperp=false) where Tpdf <: AbstractArray{Float64,2}
    # get required moments of pdf
    calculate_multipole_expansion_moments!(expansion_data,pdf,dummy_vpavperp,vpa,vperp)
    # now pass the delta f Inm to the multipole function
    calculate_rosenbluth_potential_boundary_data_multipole!(rpbd,expansion_data,vpa,vperp,
      calculate_GG=calculate_GG,calculate_dGdvperp=calculate_dGdvperp)
    return nothing
end

"""
Function to compare two instances of `RosenbluthPotentialBoundaryData` --
one assumed to contain exact results, and the other numerically computed results -- and compute
the maximum value of the error. Calls `test_boundary_data()`.
"""
function test_rosenbluth_potential_boundary_data(rpbd::RosenbluthPotentialBoundaryData,
    rpbd_exact::RosenbluthPotentialBoundaryData,vpa::FiniteElementCoordinate,
    vperp::FiniteElementCoordinate;print_to_screen=true)

    error_buffer_vpa = Array{Float64,1}(undef,vpa.n)
    error_buffer_vperp_1 = Array{Float64,1}(undef,vperp.n)
    error_buffer_vperp_2 = Array{Float64,1}(undef,vperp.n)
    max_H_err = test_boundary_data(rpbd.H_data,rpbd_exact.H_data,"H",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2,print_to_screen)
    max_dHdvpa_err = test_boundary_data(rpbd.dHdvpa_data,rpbd_exact.dHdvpa_data,"dHdvpa",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2,print_to_screen)
    max_dHdvperp_err = test_boundary_data(rpbd.dHdvperp_data,rpbd_exact.dHdvperp_data,"dHdvperp",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2,print_to_screen)
    max_G_err = test_boundary_data(rpbd.G_data,rpbd_exact.G_data,"G",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2,print_to_screen)
    max_dGdvperp_err = test_boundary_data(rpbd.dGdvperp_data,rpbd_exact.dGdvperp_data,"dGdvperp",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2,print_to_screen)
    max_d2Gdvperp2_err = test_boundary_data(rpbd.d2Gdvperp2_data,rpbd_exact.d2Gdvperp2_data,"d2Gdvperp2",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2,print_to_screen)
    max_d2Gdvperpdvpa_err = test_boundary_data(rpbd.d2Gdvperpdvpa_data,rpbd_exact.d2Gdvperpdvpa_data,"d2Gdvperpdvpa",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2,print_to_screen)
    max_d2Gdvpa2_err = test_boundary_data(rpbd.d2Gdvpa2_data,rpbd_exact.d2Gdvpa2_data,"d2Gdvpa2",vpa,vperp,error_buffer_vpa,error_buffer_vperp_1,error_buffer_vperp_2,print_to_screen)

    return max_H_err, max_dHdvpa_err, max_dHdvperp_err, max_G_err, max_dGdvperp_err, max_d2Gdvperp2_err, max_d2Gdvperpdvpa_err, max_d2Gdvpa2_err
end

"""
Function to compute the maximum error \${\\rm MAX}|f_{\\rm numerical}-f_{\\rm exact}|\$ for
instances of `VpaVperpBoundaryData`.
"""
function test_boundary_data(func::VpaVperpBoundaryData,
            func_exact::VpaVperpBoundaryData,func_name::String,
            vpa::FiniteElementCoordinate,vperp::FiniteElementCoordinate,
            buffer_vpa::Tvector, buffer_vperp_1::Tvector, buffer_vperp_2::Tvector,
            print_to_screen::Bool) where Tvector <: AbstractArray{Float64,1}
    nvpa = vpa.n
    nvperp = vperp.n
    for ivperp in 1:nvperp
        buffer_vperp_1[ivperp] = abs(func.lower_boundary_vpa[ivperp] - func_exact.lower_boundary_vpa[ivperp])
        buffer_vperp_2[ivperp] = abs(func.upper_boundary_vpa[ivperp] - func_exact.upper_boundary_vpa[ivperp])
    end
    for ivpa in 1:nvpa
        buffer_vpa[ivpa] = abs(func.upper_boundary_vperp[ivpa] - func_exact.upper_boundary_vperp[ivpa])
    end
    max_lower_vpa_err = maximum(buffer_vperp_1)
    max_upper_vpa_err = maximum(buffer_vperp_2)
    max_upper_vperp_err = maximum(buffer_vpa)
    if print_to_screen
        println(string(func_name*" boundary data:"))
        println("max(lower_vpa_err) = ",max_lower_vpa_err)
        println("max(upper_vpa_err) = ",max_upper_vpa_err)
        println("max(upper_vperp_err) = ",max_upper_vperp_err)
    end
    max_err = max(max_lower_vpa_err,max_upper_vpa_err,max_upper_vperp_err)
    return max_err
end

"""
Sets `f(vpa,vperp)` to a specied value `f_bc` at the boundaries
in `(vpa,vperp)`. `f_bc` is an instance of `VpaVperpBoundaryData`.
"""
function enforce_dirichlet_bc!(fvpavperp::Tpdf,
            vpa::FiniteElementCoordinate,
            vperp::FiniteElementCoordinate,
            f_bc::VpaVperpBoundaryData) where Tpdf <: AbstractArray{Float64,2}
    # lower vpa boundary
    for ivperp ∈ 1:vperp.n
        fvpavperp[1,ivperp] = f_bc.lower_boundary_vpa[ivperp]
    end

    # upper vpa boundary
    for ivperp ∈ 1:vperp.n
        fvpavperp[end,ivperp] = f_bc.upper_boundary_vpa[ivperp]
    end

    # upper vperp boundary
    for ivpa ∈ 1:vpa.n
        fvpavperp[ivpa,end] = f_bc.upper_boundary_vperp[ivpa]
    end
    return nothing
end

function allocate_preconditioner_matrix(vpa::FiniteElementCoordinate,
                                    vperp::FiniteElementCoordinate)
    # Assemble a 2D mass matrix in the global compound coordinate
    function identity_weak_form(ivpa, jvpa, ielement_vpa, ivperp, jvperp, ielement_vperp)
        return floor(1.0/(1.0 + abs(ivpa - jvpa) + abs(ivperp - jvperp)))
    end
    CC2D_sparse = assemble_operator(identity_weak_form, vpa, vperp, vpa.bc, vperp.bc)
    lu_obj_CC2D = lu(CC2D_sparse)
    return CC2D_sparse, lu_obj_CC2D
end

function calculate_test_particle_preconditioner!(pdf::Tpdf,
    delta_t::Float64,ms::Float64,msp::Float64,nussp::Float64,
    fkpl_arrays::FokkerPlanckBackwardEulerData;
    use_Maxwellian_Rosenbluth_coefficients=false,
    algebraic_solve_for_d2Gdvperp2=false,calculate_GG=false,
    calculate_dGdvperp=false) where Tpdf <: AbstractArray{Float64,2}

    CC2D_sparse = fkpl_arrays.CC2D_sparse
    fp_operator = fkpl_arrays.fp_operator
    vpa = fp_operator.vpa
    vperp = fp_operator.vperp
    YY_arrays = fp_operator.YY_arrays
    rosenbluth_potentials = fp_operator.rosenbluth_potentials

    # consider making a wrapper function for the following block -- repeated in fokker_planck.jl
    if use_Maxwellian_Rosenbluth_coefficients
        calculate_rosenbluth_potentials_via_analytical_Maxwellian!(rosenbluth_potentials,pdf,vpa,vperp,msp)
    else
        calculate_rosenbluth_potentials_via_elliptic_solve!(rosenbluth_potentials,pdf,
             vpa,vperp,fp_operator.fprp_solver_data,
             algebraic_solve_for_d2Gdvperp2=false,calculate_GG=false,
             calculate_dGdvperp=false)
    end
    assemble_collision_operator_preconditioner_rhs!(CC2D_sparse,
            rosenbluth_potentials,nothing,delta_t,nussp,fp_operator,1)
    # should improve on this step to avoid recreating the sparse array if possible.
    lu!(fkpl_arrays.lu_obj_CC2D, fkpl_arrays.CC2D_sparse)
    return nothing
end
function calculate_test_particle_preconditioner!(pdf::Tpdf,
    delta_t::Float64,nuref::Float64,
    fkpl_arrays::FokkerPlanckBackwardEulerData;
    use_Maxwellian_Rosenbluth_coefficients=false,
    algebraic_solve_for_d2Gdvperp2=false,calculate_GG=false,
    calculate_dGdvperp=false) where Tpdf <: AbstractArray{Float64,3}

    CC2D_sparse = fkpl_arrays.CC2D_sparse
    fp_operator = fkpl_arrays.fp_operator
    vpa = fp_operator.vpa
    vperp = fp_operator.vperp
    species = fp_operator.species
    YY_arrays = fp_operator.YY_arrays
    # dummy arrays for Rosenbluth potentials
    rosenbluth_potentials_s = fp_operator.rosenbluth_potentials_s
    rosenbluth_potentials_total = fp_operator.rosenbluth_potentials
    rosenbluth_potentials_buffer = fp_operator.rosenbluth_potentials_buffer
    # information about fixed background plasma
    fixed_background_plasma = fp_operator.fixed_background_plasma
    # information about sources and sinks
    source_data = fkpl_arrays.source_data
    # compute potentials due to evolved species
    if use_Maxwellian_Rosenbluth_coefficients
        for is in 1:species.n
            @views calculate_rosenbluth_potentials_via_analytical_Maxwellian!(
                rosenbluth_potentials_s[is],pdf[:,:,is],vpa,vperp,species.mass[is])
        end
    else
        for is in 1:species.n
            @views calculate_rosenbluth_potentials_via_elliptic_solve!(
                rosenbluth_potentials_s[is],pdf[:,:,is],vpa,vperp,fp_operator.fprp_solver_data,
                algebraic_solve_for_d2Gdvperp2=false,calculate_GG=false,
                calculate_dGdvperp=false)
        end
    end
    @inbounds begin
        # for each species, sum up the Rosenbluth potentials to make the appropriate
        # total Rosenbluth potential, and assemble the preconditioner
        for is in 1:species.n
            calculate_cross_species_rosenbluth_potential_sums!(
                    rosenbluth_potentials_total,rosenbluth_potentials_buffer,
                    rosenbluth_potentials_s,species,
                    species.zeds[is],species.mass[is],species.c0ref[is],species.u0ref[is],
                    vpa,vperp,fixed_background_plasma)
            assemble_collision_operator_preconditioner_rhs!(CC2D_sparse,
                rosenbluth_potentials_total,source_data,delta_t,nuref,fp_operator,is)
            # should improve on this step to avoid recreating the sparse array if possible.
            lu!(fkpl_arrays.lu_objs_CC2D[is], fkpl_arrays.CC2D_sparse)
        end
    end
    return nothing
end
function assemble_collision_operator_preconditioner_rhs!(CC2D_sparse::TSparseMatrix,
    rosenbluth_potentials::RosenbluthPotentialData,source_data::Union{SlowingDownSourceData,Nothing},
    delta_t::Float64,nuref::Float64,fkpl_arrays::FokkerPlanckWeakformArrays,is::Int64
    ) where TSparseMatrix <: AbstractSparseArray{Float64,Int64,2}
    # extract structs from fkpl_arrays
    # we do not extract the potentials from fkpl_arrays to permit flexibility
    # but pass this information by argument
    vpa = fkpl_arrays.vpa
    vperp = fkpl_arrays.vperp
    YY_arrays = fkpl_arrays.YY_arrays
    d2Gdvpa2 = rosenbluth_potentials.d2Gdvpa2
    d2Gdvperpdvpa = rosenbluth_potentials.d2Gdvperpdvpa
    d2Gdvperp2 = rosenbluth_potentials.d2Gdvperp2
    dHdvpa = rosenbluth_potentials.dHdvpa
    dHdvperp = rosenbluth_potentials.dHdvperp
    MMpar = YY_arrays.MMpar
    MMperp = YY_arrays.MMperp
    YYNperp = YY_arrays.YYNperp
    YYNpar = YY_arrays.YYNpar
    massfac = 2.0
    delt_nussp = delta_t*nuref
    function sink_weak_form(ivpa, jvpa, ielement_vpa,
        ivperp, jvperp, ielement_vperp,
        source_data::SlowingDownSourceData, is)
        @inbounds begin
            delt_rate = delta_t*source_data.source_input.sink_rate[is]
            vperp_range = vperp.igrid_full[1,ielement_vperp]:vperp.igrid_full[vperp.ngrid,ielement_vperp]
            vpa_range = vpa.igrid_full[1,ielement_vpa]:vpa.igrid_full[vpa.ngrid,ielement_vpa]
            @views sink_func_local = source_data.sink_func[vpa_range,vperp_range]
            @views YYNpar_local = YYNpar[:,:,jvpa,ivpa,ielement_vpa]
            @views YYNperp_local = YYNperp[:,:,jvperp,ivperp,ielement_vperp]
            result = 0.0
            for kvperp in 1:vperp.ngrid
                for kvpa in 1:vpa.ngrid
                    # first three lines represent parallel flux terms
                    # second three lines represent perpendicular flux terms
                    result += (delt_rate*YYNperp_local[1,kvperp]*
                                YYNpar_local[1,kvpa]*
                                sink_func_local[kvpa,kvperp])
                end
            end
        end
        return result
    end
    function sink_weak_form(ivpa, jvpa, ielement_vpa,
        ivperp, jvperp, ielement_vperp,
        source_data::Nothing, is)
        return 0.0
    end
    function preconditioner_weak_form(jvpa, ivpa, ielement_vpa, jvperp, ivperp, ielement_vperp)
        @inbounds begin
            # terms from I in P = I - dt * C[F,F^n]
            result = MMpar[jvpa,ivpa,ielement_vpa]*MMperp[jvperp,ivperp,ielement_vperp]
            # terms from - dt * C[F, F^n] in P
            # summation over Rosenbluth potentials
            vperp_range = vperp.igrid_full[1,ielement_vperp]:vperp.igrid_full[vperp.ngrid,ielement_vperp]
            vpa_range = vpa.igrid_full[1,ielement_vpa]:vpa.igrid_full[vpa.ngrid,ielement_vpa]
            @views d2Gdvpa2_local = d2Gdvpa2[vpa_range,vperp_range]
            @views d2Gdvperp2_local = d2Gdvperp2[vpa_range,vperp_range]
            @views d2Gdvperpdvpa_local = d2Gdvperpdvpa[vpa_range,vperp_range]
            @views dHdvpa_local = dHdvpa[vpa_range,vperp_range]
            @views dHdvperp_local = dHdvperp[vpa_range,vperp_range]
            @views YYNpar_local = YYNpar[:,:,jvpa,ivpa,ielement_vpa]
            @views YYNperp_local = YYNperp[:,:,jvperp,ivperp,ielement_vperp]
            for kvperp in 1:vperp.ngrid
                for kvpa in 1:vpa.ngrid
                    # first three lines represent parallel flux terms
                    # second three lines represent perpendicular flux terms
                    result += delt_nussp*(YYNperp_local[1,kvperp]*YYNpar_local[3,kvpa]*d2Gdvpa2_local[kvpa,kvperp] +
                                        YYNperp_local[4,kvperp]*YYNpar_local[2,kvpa]*d2Gdvperpdvpa_local[kvpa,kvperp] -
                                        massfac*YYNperp_local[1,kvperp]*YYNpar_local[2,kvpa]*dHdvpa_local[kvpa,kvperp] +
                                        # end parallel flux, start of perpendicular flux
                                        YYNperp_local[2,kvperp]*YYNpar_local[4,kvpa]*d2Gdvperpdvpa_local[kvpa,kvperp] +
                                        YYNperp_local[3,kvperp]*YYNpar_local[1,kvpa]*d2Gdvperp2_local[kvpa,kvperp] -
                                        massfac*YYNperp_local[2,kvperp]*YYNpar_local[1,kvpa]*dHdvperp_local[kvpa,kvperp])
                end
            end
            result += sink_weak_form(ivpa, jvpa, ielement_vpa, ivperp, jvperp, ielement_vperp, source_data, is)
        end
        return result
    end

    # should improve on this step to avoid recreating the sparse array if possible.
    CC2D_sparse .= assemble_operator(preconditioner_weak_form, vpa, vperp, vpa.bc, vperp.bc)
    return nothing
end

function advance_linearised_test_particle_collisions!(
        pdf::Tpdf, fkpl_arrays::FokkerPlanckBackwardEulerData
        ) where Tpdf <: AbstractArray{Float64,2}
    # (the LU decomposition object for)
    # the backward Euler time advance matrix
    # for linearised test particle collisions K * dF = C[dF, F^n+1].
    # this is also the LU decomposition of the approximate Jacobian
    # for the nonlinear residual R = F^n+1 - F^n - C[F^n+1, F^n+1]
    lu_CC = fkpl_arrays.lu_obj_CC2D
    advance_linearised_test_particle_collisions!(pdf,fkpl_arrays.fp_operator,lu_CC)
    return nothing
end
function advance_linearised_test_particle_collisions!(
            pdf::Tpdf, fkpl_arrays::FokkerPlanckWeakformArrays,
            lu_CC::SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}
            ) where Tpdf <: AbstractArray{Float64,2}
    # lu_CC (the LU decomposition object for)
    # the backward Euler time advance matrix
    # for linearised test particle collisions K * dF = C[dF, F^n+1].
    # this is also the LU decomposition of the approximate Jacobian
    # for the nonlinear residual R = F^n+1 - F^n - C[F^n+1, F^n+1]
    vpa = fkpl_arrays.vpa
    vperp = fkpl_arrays.vperp
    # function to solve K * F^n+1 = M * F^n
    # and return F^n+1 in place in pdf
    # enforce zero BCs on pdf in so that
    # these BCs are imposed via the unit boundary
    # values in CC2D_sparse, in the event BCs are used
    enforce_vpavperp_BCs!(pdf,vpa,vperp)
    # extra dummy arrays
    pdf_scratch = fkpl_arrays.fprp_solver_data.matrix_operators.rhsvpavperp
    pdf_dummy = fkpl_arrays.fprp_solver_data.matrix_operators.S_dummy
    # mass matrix for RHS
    MM2D_sparse = fkpl_arrays.fprp_solver_data.matrix_operators.MM2D_sparse
    @views @. pdf_scratch = pdf
    pdf_c = vec(pdf)
    pdf_scratch_c = vec(pdf_scratch)
    pdf_dummy_c = vec(pdf_dummy)
    mul!(pdf_dummy_c, MM2D_sparse, pdf_scratch_c)
    ldiv!(pdf_c,lu_CC,pdf_dummy_c)
    return nothing
end
function advance_linearised_test_particle_collisions!(
            pdf::Tpdf, fkpl_arrays::FokkerPlanckBackwardEulerData
            ) where Tpdf <: AbstractArray{Float64,3}
    # (the vector of LU decomposition objects for)
    # the backward Euler time advance matrix
    # for multi-species linearised test particle collisions K * dF = C[dF, F^n+1].
    # this is also the LU decomposition of the approximate Jacobian
    # for the nonlinear residual R = F^n+1 - F^n - C[F^n+1, F^n+1]
    lu_CC = fkpl_arrays.lu_objs_CC2D
    species = fkpl_arrays.fp_operator.species
    for is in 1:species.n
        @views advance_linearised_test_particle_collisions!(pdf[:,:,is],fkpl_arrays.fp_operator,lu_CC[is])
    end
    return nothing
end

"""
Function to assemble the RHS of the kinetic equation due to the collision operator,
in weak form. Once the array `rhsvpavperp` contains the assembled weak-form collision operator,
a mass matrix solve still must be carried out to find the time derivative of the distribution function
due to collisions.
"""
function assemble_explicit_collision_operator_rhs_serial!(
    rhsvpavperp::Array{Float64,2},pdfs::Tpdf,
    rosenbluth_potentials_sp::RosenbluthPotentialData,ms::Float64,msp::Float64,nussp::Float64,
    vpa::FiniteElementCoordinate,vperp::FiniteElementCoordinate,
    YY_arrays::CollisionOperatorArrays
    ) where Tpdf <: AbstractArray{Float64,2}
    d2Gspdvpa2 = rosenbluth_potentials_sp.d2Gdvpa2
    d2Gspdvperpdvpa = rosenbluth_potentials_sp.d2Gdvperpdvpa
    d2Gspdvperp2 = rosenbluth_potentials_sp.d2Gdvperp2
    dHspdvpa = rosenbluth_potentials_sp.dHdvpa
    dHspdvperp = rosenbluth_potentials_sp.dHdvperp
    @inbounds begin
        # assemble RHS of collision operator
        rhsc = vec(rhsvpavperp)
        @. rhsc = 0.0
        massfac = 2.0*(ms/msp) # extracted to reduce * and / operators in hot loop
        # loop over elements
        for ielement_vperp in 1:vperp.nelement
            @views YYNperp = YY_arrays.YYNperp[:,:,:,:,ielement_vperp]
            @views vperp_igrid_full = vperp.igrid_full[:,ielement_vperp]
            imin_vperp, imax_vperp = vperp_igrid_full[1], vperp_igrid_full[vperp.ngrid]
            for ielement_vpa in 1:vpa.nelement
                @views YYNpar = YY_arrays.YYNpar[:,:,:,:,ielement_vpa]
                @views vpa_igrid_full = vpa.igrid_full[:,ielement_vpa]
                imin_vpa, imax_vpa = vpa_igrid_full[1], vpa_igrid_full[vpa.ngrid]
                @views d2Gspdvpa2_local = d2Gspdvpa2[imin_vpa:imax_vpa,imin_vperp:imax_vperp]
                @views d2Gspdvperp2_local = d2Gspdvperp2[imin_vpa:imax_vpa,imin_vperp:imax_vperp]
                @views d2Gspdvperpdvpa_local = d2Gspdvperpdvpa[imin_vpa:imax_vpa,imin_vperp:imax_vperp]
                @views dHspdvpa_local = dHspdvpa[imin_vpa:imax_vpa,imin_vperp:imax_vperp]
                @views dHspdvperp_local = dHspdvperp[imin_vpa:imax_vpa,imin_vperp:imax_vperp]
                @views pdfs_local = pdfs[imin_vpa:imax_vpa,imin_vperp:imax_vperp]
                # loop over field positions in each element
                for ivperp_local in 1:vperp.ngrid
                    ivperp_global = vperp_igrid_full[ivperp_local]
                    for ivpa_local in 1:vpa.ngrid
                        ivpa_global = vpa_igrid_full[ivpa_local]
                        # global compound index
                        ic_global = ic_func(ivpa_global,ivperp_global,vpa.n)
                        # carry out the matrix sum on each 2D element
                        result = 0.0
                        for jvperpp_local in 1:vperp.ngrid
                            for kvperpp_local in 1:vperp.ngrid
                                @views YYNperp_kji = YYNperp[:,kvperpp_local,jvperpp_local,ivperp_local]
                                for jvpap_local in 1:vpa.ngrid
                                    pdfjj = pdfs_local[jvpap_local,jvperpp_local]
                                    for kvpap_local in 1:vpa.ngrid
                                        @views YYNpar_kji = YYNpar[:,kvpap_local,jvpap_local,ivpa_local]
                                        d2Gspdvperpdvpa_kk = d2Gspdvperpdvpa_local[kvpap_local,kvperpp_local]
                                        # first three lines represent parallel flux terms
                                        # second three lines represent perpendicular flux terms
                                        result += pdfjj*(YYNperp_kji[1]*YYNpar_kji[3]*d2Gspdvpa2_local[kvpap_local,kvperpp_local] +
                                                            YYNperp_kji[4]*YYNpar_kji[2]*d2Gspdvperpdvpa_kk -
                                                            massfac*YYNperp_kji[1]*YYNpar_kji[2]*dHspdvpa_local[kvpap_local,kvperpp_local] +
                                                            # end parallel flux, start of perpendicular flux
                                                            YYNperp_kji[2]*YYNpar_kji[4]*d2Gspdvperpdvpa_kk +
                                                            YYNperp_kji[3]*YYNpar_kji[1]*d2Gspdvperp2_local[kvpap_local,kvperpp_local] -
                                                            massfac*YYNperp_kji[2]*YYNpar_kji[1]*dHspdvperp_local[kvpap_local,kvperpp_local])
                                        # sum = YYNperp_kji[1]*YYNpar_kji[3]*d2Gspdvpa2_local[kvpap_local,kvperpp_local]
                                        # sum += YYNperp_kji[4]*YYNpar_kji[2]*d2Gspdvperpdvpa_kk
                                        # sum -= massfac*YYNperp_kji[1]*YYNpar_kji[2]*dHspdvpa_local[kvpap_local,kvperpp_local]
                                        #                     # end parallel flux, start of perpendicular flux
                                        # sum += YYNperp_kji[2]*YYNpar_kji[4]*d2Gspdvperpdvpa_kk
                                        # sum += YYNperp_kji[3]*YYNpar_kji[1]*d2Gspdvperp2_local[kvpap_local,kvperpp_local]
                                        # sum -= massfac*YYNperp_kji[2]*YYNpar_kji[1]*dHspdvperp_local[kvpap_local,kvperpp_local]
                                        # sum *= pdfjj
                                        # result += sum
                                    end
                                end
                            end
                        end
                        rhsc[ic_global] += result
                    end
                end
            end
        end
        # correct for minus sign due to integration by parts
        # and multiply by the normalised collision frequency
        @. rhsc *= -nussp
        return nothing
    end
end

"""
Elliptic solve function.

    field: the solution
    source: the source function on the RHS
    boundary data: the known values of field at infinity
    lu_object_lhs: the object for the differential operator that defines field
    matrix_rhs: the weak matrix acting on the source vector
    vpa, vperp: coordinate structs

Note: all variants of `elliptic_solve!()` run only in serial. They do not handle
shared-memory parallelism themselves. The calling site must ensure that
`elliptic_solve!()` is only called by one process in a shared-memory block.
"""
function elliptic_solve!(field::Tpdf1,source::Tpdf2,
            boundary_data::VpaVperpBoundaryData,
            lu_object_lhs::SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64},
            matrix_rhs::TSparseMatrix,rhsvpavperp::Tpdf2,
            vpa::FiniteElementCoordinate,
            vperp::FiniteElementCoordinate) where {Tpdf1 <: AbstractArray{Float64,2},
                Tpdf2 <: AbstractArray{Float64,2},
                TSparseMatrix <: AbstractSparseArray{Float64,Int64,2}}
    @inbounds begin
        # assemble the rhs of the weak system

        # get data into the compound index format
        sc = vec(source)
        fc = vec(field)
        rhsc = vec(rhsvpavperp)
        mul!(rhsc,matrix_rhs,sc)
        # enforce the boundary conditions
        enforce_dirichlet_bc!(rhsvpavperp,vpa,vperp,boundary_data)
        # solve the linear system
        ldiv!(fc, lu_object_lhs, rhsc)

        return nothing
    end
end
# same as above but source is made of two different terms
# with different weak matrices
function elliptic_solve!(field::Tpdf1,source_1::Tpdf2,source_2::Tpdf2,
            boundary_data::VpaVperpBoundaryData,
            lu_object_lhs::SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64},
            matrix_rhs_1::TSparseMatrix, matrix_rhs_2::TSparseMatrix,
            rhs::Tpdf2,vpa::FiniteElementCoordinate,
            vperp::FiniteElementCoordinate
            ) where {Tpdf1 <: AbstractArray{Float64,2},
                Tpdf2 <: AbstractArray{Float64,2},
                TSparseMatrix <: AbstractSparseArray{Float64,Int64,2}}

    @inbounds begin
        # assemble the rhs of the weak system

        # get data into the compound index format
        sc_1 = vec(source_1)
        sc_2 = vec(source_2)
        rhsc = vec(rhs)
        fc = vec(field)

        # Do  rhsc = matrix_rhs_1*sc_1
        mul!(rhsc, matrix_rhs_1, sc_1)

        # Do rhsc = matrix_rhs_2*sc_2 + rhsc
        mul!(rhsc, matrix_rhs_2, sc_2, 1.0, 1.0)

        # enforce the boundary conditions
        enforce_dirichlet_bc!(rhs,vpa,vperp,boundary_data)
        # solve the linear system
        ldiv!(fc, lu_object_lhs, rhsc)

        return nothing
    end
end

"""
Same as `elliptic_solve!()` above but no Dirichlet boundary conditions are imposed,
because the function is only used where the `lu_object_lhs` is derived from a mass matrix.
The source is made of two different terms with different weak matrices
because of the form of the only algebraic equation that we consider.

Note: `algebraic_solve!()` run only in serial. They do not handle shared-memory
parallelism themselves. The calling site must ensure that `algebraic_solve!()` is only
called by one process in a shared-memory block.
"""
function algebraic_solve!(field::Tpdf,source_1::Tpdf,source_2::Tpdf,
            boundary_data::VpaVperpBoundaryData,
            lu_object_lhs::SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64},
            matrix_rhs_1::TSparseMatrix, matrix_rhs_2::TSparseMatrix,
            rhs::Tpdf,vpa::FiniteElementCoordinate,
            vperp::FiniteElementCoordinate
            ) where {Tpdf <: AbstractArray{Float64,2}, TSparseMatrix <: AbstractSparseArray{Float64,Int64,2}}

    @inbounds begin
        # assemble the rhs of the weak system

        # get data into the compound index format
        sc_1 = vec(source_1)
        sc_2 = vec(source_2)
        rhsc = vec(rhs)
        fc = vec(field)

        # Do  rhsc = matrix_rhs_1*sc_1
        mul!(rhsc, matrix_rhs_1, sc_1)

        # Do rhsc = matrix_rhs_2*sc_2 + rhsc
        mul!(rhsc, matrix_rhs_2, sc_2, 1.0, 1.0)

        # solve the linear system
        ldiv!(fc, lu_object_lhs, rhsc)

        return nothing
    end
end

"""
Function to solve the appropriate elliptic PDEs to find the
Rosenbluth potentials. First, we calculate the Rosenbluth potentials
at the boundary with the direct integration method. Then, we use this
data to solve the elliptic PDEs with the boundary data providing an
accurate Dirichlet boundary condition on the maximum `vpa` and `vperp`
of the domain. We use the sparse LU decomposition from the LinearAlgebra package
to solve the PDE matrix equations.
"""
function calculate_rosenbluth_potentials_via_elliptic_solve!(
             rosenbluth_potentials::RosenbluthPotentialData,ffsp_in::Tpdf,
             vpa::FiniteElementCoordinate,vperp::FiniteElementCoordinate,
             fkpl_arrays::FokkerPlanckRosenbluthPotentialSolverData;
             algebraic_solve_for_d2Gdvperp2=false,calculate_GG=false,
             calculate_dGdvperp=false) where Tpdf <: AbstractArray{Float64,2}
    GG = rosenbluth_potentials.GG
    HH = rosenbluth_potentials.HH
    dHdvpa = rosenbluth_potentials.dHdvpa
    dHdvperp = rosenbluth_potentials.dHdvperp
    dGdvperp = rosenbluth_potentials.dGdvperp
    d2Gdvperp2 = rosenbluth_potentials.d2Gdvperp2
    d2Gdvpa2 = rosenbluth_potentials.d2Gdvpa2
    d2Gdvperpdvpa = rosenbluth_potentials.d2Gdvperpdvpa
    multipole_expansion_moments = rosenbluth_potentials.multipole_expansion_moments
    # extract the necessary precalculated and buffer arrays from fokkerplanck_arrays
    matrix_operators = fkpl_arrays.matrix_operators
    MM2D_sparse = matrix_operators.MM2D_sparse
    KKpar2D_sparse = matrix_operators.KKpar2D_sparse
    KKperp2D_sparse = matrix_operators.KKperp2D_sparse
    LP2D_sparse = matrix_operators.LP2D_sparse
    LV2D_sparse = matrix_operators.LV2D_sparse
    PUperp2D_sparse = matrix_operators.PUperp2D_sparse
    PPparPUperp2D_sparse = matrix_operators.PPparPUperp2D_sparse
    PPpar2D_sparse = matrix_operators.PPpar2D_sparse
    MMparMNperp2D_sparse = matrix_operators.MMparMNperp2D_sparse
    KPperp2D_sparse = matrix_operators.KPperp2D_sparse
    lu_obj_MM = matrix_operators.lu_obj_MM
    lu_obj_LP = matrix_operators.lu_obj_LP
    lu_obj_LV = matrix_operators.lu_obj_LV
    lu_obj_LB = matrix_operators.lu_obj_LB

    bwgt = fkpl_arrays.bwgt
    rpbd = fkpl_arrays.rpbd

    S_dummy = matrix_operators.S_dummy
    Q_dummy = matrix_operators.Q_dummy
    rhsvpavperp = matrix_operators.rhsvpavperp
    boundary_data_option = fkpl_arrays.boundary_data_option
    # calculate the boundary data
    if boundary_data_option == multipole_expansion
        calculate_rosenbluth_potential_boundary_data_multipole!(rpbd,multipole_expansion_moments,ffsp_in,vpa,vperp,
          calculate_GG=calculate_GG,calculate_dGdvperp=(calculate_dGdvperp||algebraic_solve_for_d2Gdvperp2))
    elseif boundary_data_option == delta_f_multipole # use a variant of the multipole method
        calculate_rosenbluth_potential_boundary_data_delta_f_multipole!(rpbd,multipole_expansion_moments,ffsp_in,S_dummy,vpa,vperp,
          calculate_GG=calculate_GG,calculate_dGdvperp=(calculate_dGdvperp||algebraic_solve_for_d2Gdvperp2))
    elseif boundary_data_option == direct_integration  # use direct integration on the boundary
        calculate_rosenbluth_potential_boundary_data!(rpbd,bwgt,ffsp_in,vpa,vperp,
         calculate_GG=calculate_GG,calculate_dGdvperp=(calculate_dGdvperp||algebraic_solve_for_d2Gdvperp2))
    else
        error("No valid boundary_data_option specified. \n
              Pick  boundary_data_option='$multipole_expansion' \n
              or  boundary_data_option='$delta_f_multipole' \n
              or boundary_data_option='$direct_integration'")
    end
    # carry out the elliptic solves required
    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                S_dummy[ivpa,ivperp] = -(4.0*pi)*ffsp_in[ivpa,ivperp]
            end
        end
    end

    elliptic_solve!(HH, S_dummy, rpbd.H_data, lu_obj_LP, MM2D_sparse, rhsvpavperp,
                    vpa, vperp)
    elliptic_solve!(dHdvpa, S_dummy, rpbd.dHdvpa_data, lu_obj_LP, PPpar2D_sparse,
                    rhsvpavperp, vpa, vperp)
    elliptic_solve!(dHdvperp, S_dummy, rpbd.dHdvperp_data, lu_obj_LV, PUperp2D_sparse,
                    rhsvpavperp, vpa, vperp)

    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                S_dummy[ivpa,ivperp] = 2.0*HH[ivpa,ivperp]
            end
        end
    end

    if calculate_GG
        elliptic_solve!(GG, S_dummy, rpbd.G_data, lu_obj_LP, MM2D_sparse,
                        rhsvpavperp, vpa, vperp)
    end
    if calculate_dGdvperp || algebraic_solve_for_d2Gdvperp2
        elliptic_solve!(dGdvperp, S_dummy, rpbd.dGdvperp_data, lu_obj_LV,
                        PUperp2D_sparse, rhsvpavperp, vpa, vperp)
    end
    elliptic_solve!(d2Gdvpa2, S_dummy, rpbd.d2Gdvpa2_data, lu_obj_LP, KKpar2D_sparse,
                    rhsvpavperp, vpa, vperp)
    elliptic_solve!(d2Gdvperpdvpa, S_dummy, rpbd.d2Gdvperpdvpa_data, lu_obj_LV,
                    PPparPUperp2D_sparse, rhsvpavperp, vpa, vperp)

    if algebraic_solve_for_d2Gdvperp2
        @inbounds begin
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    S_dummy[ivpa,ivperp] = 2.0*HH[ivpa,ivperp] - d2Gdvpa2[ivpa,ivperp]
                    Q_dummy[ivpa,ivperp] = -dGdvperp[ivpa,ivperp]
                end
            end
        end
        # use the algebraic solve function to find
        # d2Gdvperp2 = 2H - d2Gdvpa2 - (1/vperp)dGdvperp
        # using a weak form
        algebraic_solve!(d2Gdvperp2, S_dummy, Q_dummy, rpbd.d2Gdvperp2_data,
                            lu_obj_MM, MM2D_sparse, MMparMNperp2D_sparse, rhsvpavperp,
                            vpa, vperp)
    else
        # solve a weak-form PDE for d2Gdvperp2
        @inbounds begin
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    #S_dummy[ivpa,ivperp] = 2.0*HH[ivpa,ivperp] # <- this is already the value of
                                                                #    S_dummy calculated above
                    Q_dummy[ivpa,ivperp] = 2.0*d2Gdvpa2[ivpa,ivperp]
                end
            end
        end
        elliptic_solve!(d2Gdvperp2, S_dummy, Q_dummy, rpbd.d2Gdvperp2_data, lu_obj_LB,
                        KPperp2D_sparse, MMparMNperp2D_sparse, rhsvpavperp, vpa,
                        vperp)
    end
    return nothing
end

"""
Function to calculate Rosenbluth potentials in the entire
domain of `(vpa,vperp)` by direct integration.
"""

function calculate_rosenbluth_potentials_via_direct_integration!(GG::Tpdf1,
             HH::Tpdf1,dHdvpa::Tpdf1,dHdvperp::Tpdf1,
             d2Gdvpa2::Tpdf1,dGdvperp::Tpdf1,d2Gdvperpdvpa::Tpdf1,d2Gdvperp2::Tpdf1,
             ffsp_in::Tpdf2,
             vpa::FiniteElementCoordinate,vperp::FiniteElementCoordinate,
             fkpl_arrays::FokkerPlanckArraysDirectIntegration
             ) where {Tpdf1 <: AbstractArray{Float64,2}, Tpdf2 <: AbstractArray{Float64,2}}
    dfdvpa = fkpl_arrays.dfdvpa
    dfdvperp = fkpl_arrays.dfdvperp
    d2fdvperpdvpa = fkpl_arrays.d2fdvperpdvpa
    G0_weights = fkpl_arrays.G0_weights
    G1_weights = fkpl_arrays.G1_weights
    H0_weights = fkpl_arrays.H0_weights
    H1_weights = fkpl_arrays.H1_weights
    H2_weights = fkpl_arrays.H2_weights
    H3_weights = fkpl_arrays.H3_weights
    # first compute the derivatives of fs' (the integration weights assume d fs' dvpa and d fs' dvperp are known)
    @inbounds for ivperp in 1:vperp.n
        @views first_derivative!(dfdvpa[:,ivperp], ffsp_in[:,ivperp], vpa)
    end
    @inbounds for ivpa in 1:vpa.n
        @views first_derivative!(dfdvperp[ivpa,:], ffsp_in[ivpa,:], vperp)
        @views first_derivative!(d2fdvperpdvpa[ivpa,:], dfdvpa[ivpa,:], vperp)
    end
    # with the integrands calculated, compute the integrals
    calculate_rosenbluth_integrals!(GG,d2Gdvpa2,dGdvperp,d2Gdvperpdvpa,
                                        d2Gdvperp2,HH,dHdvpa,dHdvperp,
                                        ffsp_in,dfdvpa,dfdvperp,d2fdvperpdvpa,
                                        G0_weights,G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                                        vpa.n,vperp.n)
    return nothing
end

"""
Function to calculate Rosenbluth potentials for shifted Maxwellians
using an analytical specification
"""
function calculate_rosenbluth_potentials_via_analytical_Maxwellian!(
    rosenbluth_potentials::RosenbluthPotentialData,
    ffsp_in::Tpdf,vpa::FiniteElementCoordinate,
    vperp::FiniteElementCoordinate,mass::Float64) where Tpdf <: AbstractArray{Float64,2}
    dens = get_density(ffsp_in, vpa, vperp)
    upar = get_upar(ffsp_in, vpa, vperp, dens)
    pressure = get_pressure(ffsp_in, vpa, vperp, upar, mass)
    vth = sqrt(2.0*pressure/(dens*mass))
    calculate_rosenbluth_potentials_via_analytical_Maxwellian!(
        rosenbluth_potentials,dens,upar,vth,vpa,vperp)
    return nothing
end
function calculate_rosenbluth_potentials_via_analytical_Maxwellian!(
    rosenbluth_potentials::RosenbluthPotentialData,
    dens::Float64,upar::Float64,vth::Float64,
    vpa::FiniteElementCoordinate,vperp::FiniteElementCoordinate)
    GG = rosenbluth_potentials.GG
    HH = rosenbluth_potentials.HH
    dHdvpa = rosenbluth_potentials.dHdvpa
    dHdvperp = rosenbluth_potentials.dHdvperp
    dGdvperp = rosenbluth_potentials.dGdvperp
    d2Gdvperp2 = rosenbluth_potentials.d2Gdvperp2
    d2Gdvpa2 = rosenbluth_potentials.d2Gdvpa2
    d2Gdvperpdvpa = rosenbluth_potentials.d2Gdvperpdvpa
    expansion_data = rosenbluth_potentials.multipole_expansion_moments
    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                GG[ivpa,ivperp] = G_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
                HH[ivpa,ivperp] = H_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
                d2Gdvpa2[ivpa,ivperp] = d2Gdvpa2_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
                d2Gdvperp2[ivpa,ivperp] = d2Gdvperp2_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
                dGdvperp[ivpa,ivperp] = dGdvperp_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
                d2Gdvperpdvpa[ivpa,ivperp] = d2Gdvperpdvpa_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
                dHdvpa[ivpa,ivperp] = dHdvpa_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
                dHdvperp[ivpa,ivperp] = dHdvperp_Maxwellian(dens,upar,vth,vpa.grid[ivpa],vperp.grid[ivperp])
            end
        end
    end
    calculate_analytical_Maxwellian_multipole_expansion_moments!(expansion_data,dens,upar,vth)
    return nothing
end
function calculate_analytical_Maxwellian_multipole_expansion_moments!(expansion_data::Nothing,
            dens::Float64,upar::Float64,vth::Float64)
    # do nothing, we do not use multipole expansion for direct_integration option.
    return nothing
end
function calculate_analytical_Maxwellian_multipole_expansion_moments!(expansion_data::DeltaFMultipoleMoments,
            dens::Float64,upar::Float64,vth::Float64)
    expansion_data.Maxwellian_moments .= [dens,upar,vth]
    # set Inm to zero because moments of F captured in Maxwellian moments and the
    # Rosenbluth potential formulae for Maxwellian distributions
    expansion_data.Inm_vec .= 0.0
    return nothing
end
function calculate_analytical_Maxwellian_multipole_expansion_moments!(Inm_vec::Vector{Float64},
            dens::Float64,upar::Float64,vth::Float64)
    I00 = FMaxwell_moment(0,0,dens,upar,vth)
    I10 = FMaxwell_moment(1,0,dens,upar,vth)
    I20 = FMaxwell_moment(2,0,dens,upar,vth)
    I30 = FMaxwell_moment(3,0,dens,upar,vth)
    I40 = FMaxwell_moment(4,0,dens,upar,vth)
    I50 = FMaxwell_moment(5,0,dens,upar,vth)
    I60 = FMaxwell_moment(6,0,dens,upar,vth)
    I70 = FMaxwell_moment(7,0,dens,upar,vth)
    I80 = FMaxwell_moment(8,0,dens,upar,vth)

    I02 = FMaxwell_moment(0,2,dens,upar,vth)
    I12 = FMaxwell_moment(1,2,dens,upar,vth)
    I22 = FMaxwell_moment(2,2,dens,upar,vth)
    I32 = FMaxwell_moment(3,2,dens,upar,vth)
    I42 = FMaxwell_moment(4,2,dens,upar,vth)
    I52 = FMaxwell_moment(5,2,dens,upar,vth)
    I62 = FMaxwell_moment(6,2,dens,upar,vth)

    I04 = FMaxwell_moment(0,4,dens,upar,vth)
    I14 = FMaxwell_moment(1,4,dens,upar,vth)
    I24 = FMaxwell_moment(2,4,dens,upar,vth)
    I34 = FMaxwell_moment(3,4,dens,upar,vth)
    I44 = FMaxwell_moment(4,4,dens,upar,vth)

    I06 = FMaxwell_moment(0,6,dens,upar,vth)
    I16 = FMaxwell_moment(1,6,dens,upar,vth)
    I26 = FMaxwell_moment(2,6,dens,upar,vth)

    I08 = FMaxwell_moment(0,8,dens,upar,vth)

    Inm_vec .= [I00, I10, I20, I30, I40, I50, I60, I70, I80,
                    I02, I12, I22, I32, I42, I52, I62,
                    I04, I14, I24, I34, I44,
                    I06, I16, I26,
                    I08]
    return nothing
end
function FMaxwell_moment(n::Int64,m::Int64,
            dens::Float64,upar::Float64,vth::Float64)
    prefactor = dens*vth^(m+n)
    Im = factorial(Int(m/2)) # perpendicular integral I_m = 2 int^infty_0 x^{m+1} exp(-x^2) d x
    In = 0.0 # parallel integral 1/sqrt(pi) * int^infty_-infty (y + upar/vth)^n exp(-y^2) d y
    for j in 0:n
        if mod(j,2) == 0 # J_j = 0 for odd j
            if j > 0
                Jj = (2.0^(-j+1))*factorial(j-1)/factorial(Int(j/2)-1) # J_j = int^infty_-infty y^(j) exp(-y^2) d y / sqrt(pi)
            else
                Jj = 1.0
            end
            In += binomial(n,j)*Jj*((upar/vth)^(n-j))
        end
    end
    return prefactor*Im*In
end
"""
Function to carry out the integration of the revelant
distribution functions to form the required coefficients
for the full-F operator. We assume that the weights are
precalculated. The function takes as arguments the arrays
of coefficients (which we fill), the required distributions,
the precomputed weights, the indicies of the `field' velocities,
and the sizes of the primed vpa and vperp coordinates arrays.
"""
function calculate_rosenbluth_integrals!(GG::Tpdf,d2Gspdvpa2::Tpdf,dGspdvperp::Tpdf,d2Gspdvperpdvpa::Tpdf,
                                        d2Gspdvperp2::Tpdf,HH::Tpdf,dHspdvpa::Tpdf,dHspdvperp::Tpdf,
                                        fsp::Tpdf,dfspdvpa::Tpdf,dfspdvperp::Tpdf,d2fspdvperpdvpa::Tpdf,
                                        G0_weights::Twgts,G1_weights::Twgts,H0_weights::Twgts,
                                        H1_weights::Twgts,H2_weights::Twgts,H3_weights::Twgts,
                                        nvpa::Int64,nvperp::Int64) where {Tpdf <: AbstractArray{Float64,2},
                                                                            Twgts <: AbstractArray{Float64,4}}
    @inbounds begin
        for ivperp in 1:nvperp
            for ivpa in 1:nvpa
                GG[ivpa,ivperp] = 0.0
                d2Gspdvpa2[ivpa,ivperp] = 0.0
                dGspdvperp[ivpa,ivperp] = 0.0
                d2Gspdvperpdvpa[ivpa,ivperp] = 0.0
                d2Gspdvperp2[ivpa,ivperp] = 0.0
                HH[ivpa,ivperp] = 0.0
                dHspdvpa[ivpa,ivperp] = 0.0
                dHspdvperp[ivpa,ivperp] = 0.0
                for ivperpp in 1:nvperp
                    for ivpap in 1:nvpa
                        GG[ivpa,ivperp] += G0_weights[ivpap,ivperpp,ivpa,ivperp]*fsp[ivpap,ivperpp]
                        #d2Gspdvpa2[ivpa,ivperp] += G0_weights[ivpap,ivperpp,ivpa,ivperp]*d2fspdvpa2[ivpap,ivperpp]
                        d2Gspdvpa2[ivpa,ivperp] += H3_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvpa[ivpap,ivperpp]
                        dGspdvperp[ivpa,ivperp] += G1_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvperp[ivpap,ivperpp]
                        d2Gspdvperpdvpa[ivpa,ivperp] += G1_weights[ivpap,ivperpp,ivpa,ivperp]*d2fspdvperpdvpa[ivpap,ivperpp]
                        #d2Gspdvperp2[ivpa,ivperp] += G2_weights[ivpap,ivperpp,ivpa,ivperp]*d2fspdvperp2[ivpap,ivperpp] + G3_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvperp[ivpap,ivperpp]
                        d2Gspdvperp2[ivpa,ivperp] += H2_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvperp[ivpap,ivperpp]
                        HH[ivpa,ivperp] += H0_weights[ivpap,ivperpp,ivpa,ivperp]*fsp[ivpap,ivperpp]
                        dHspdvpa[ivpa,ivperp] += H0_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvpa[ivpap,ivperpp]
                        dHspdvperp[ivpa,ivperp] += H1_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvperp[ivpap,ivperpp]
                    end
                end
            end
        end
    end
    return nothing
end

"""
Function to enforce non-natural boundary conditions on the collision operator
result to be consistent with the boundary conditions imposed on the
distribution function.
"""
function enforce_vpavperp_BCs!(pdf::Tpdf,
            vpa::FiniteElementCoordinate,
            vperp::FiniteElementCoordinate) where Tpdf <: AbstractArray{Float64,2}
    nvpa = vpa.n
    nvperp = vperp.n
    # vpa non-natural boundary conditions
    # zero at infinity
    if vpa.bc == zero_boundary_condition
        @inbounds for ivperp in 1:vperp.n
            pdf[1,ivperp] = 0.0
            pdf[nvpa,ivperp] = 0.0
        end
    end
    # vperp non-natural boundary conditions
    # zero boundary condition at infinity
    if vperp.bc == zero_boundary_condition
        @inbounds for ivpa in 1:vpa.n
            pdf[ivpa,nvperp] = 0.0
        end
    end
    return nothing
end

"""
"""
function calculate_cross_species_rosenbluth_potential_sums!(
                rosenbluth_potentials::RosenbluthPotentialData,
                rosenbluth_potentials_buffer::RosenbluthPotentialData,
                Zs::Float64,ms::Float64,c0refs::Float64,u0refs::Float64,
                vpa::FiniteElementCoordinate,vperp::FiniteElementCoordinate,
                fixed_background_plasma::Union{Nothing,FixedBackgroundPlasmaData})
    calculate_cross_species_rosenbluth_potential_sums!(
                rosenbluth_potentials,rosenbluth_potentials_buffer,
                nothing,nothing,Zs,ms,c0refs,u0refs,vpa,vperp,
                fixed_background_plasma)
    return nothing
end
function calculate_cross_species_rosenbluth_potential_sums!(
                rosenbluth_potentials_total::RosenbluthPotentialData,
                rosenbluth_potentials_buffer::RosenbluthPotentialData,
                rosenbluth_potentials_s::Union{Nothing,Vector{RosenbluthPotentialData}},
                species::Union{Nothing,SpeciesData},
                Zs::Float64,ms::Float64,c0refs::Float64,u0refs::Float64,
                vpa::FiniteElementCoordinate,vperp::FiniteElementCoordinate,
                fixed_background_plasma::Union{Nothing,FixedBackgroundPlasmaData})
    # zero Rosenbluth potentials before summation
    dHdvpa = rosenbluth_potentials_total.dHdvpa
    dHdvperp = rosenbluth_potentials_total.dHdvperp
    d2Gdvperp2 = rosenbluth_potentials_total.d2Gdvperp2
    d2Gdvpa2 = rosenbluth_potentials_total.d2Gdvpa2
    d2Gdvperpdvpa = rosenbluth_potentials_total.d2Gdvperpdvpa
    d2Gdvpa2 .= 0.0
    d2Gdvperpdvpa .= 0.0
    d2Gdvperp2 .= 0.0
    dHdvpa .= 0.0
    dHdvperp .= 0.0
    # sum potentials from the evolved species
    sum_cross_species_rosenbluth_potentials!(
                rosenbluth_potentials_total,
                rosenbluth_potentials_buffer,
                rosenbluth_potentials_s,
                species,Zs,ms,c0refs,u0refs,vpa,vperp)
    # sum potentials from the fixed (unevolved) species
    sum_cross_species_rosenbluth_potentials!(
                rosenbluth_potentials_total,
                rosenbluth_potentials_buffer,
                fixed_background_plasma,
                Zs,ms,c0refs,u0refs,vpa,vperp)
    return nothing
end
function sum_cross_species_rosenbluth_potentials!(
                rosenbluth_potentials::RosenbluthPotentialData,
                rosenbluth_potentials_buffer::RosenbluthPotentialData,
                fixed_background_plasma::FixedBackgroundPlasmaData,
                Zs::Float64,ms::Float64,c0refs::Float64,u0refs::Float64,
                vpa::FiniteElementCoordinate,vperp::FiniteElementCoordinate)
    sum_cross_species_rosenbluth_potentials!(
                rosenbluth_potentials,
                rosenbluth_potentials_buffer,
                fixed_background_plasma.rosenbluth_potentials_s,
                fixed_background_plasma.species,Zs,ms,c0refs,u0refs,vpa,vperp)
    return nothing
end
function sum_cross_species_rosenbluth_potentials!(
                rosenbluth_potentials::RosenbluthPotentialData,
                rosenbluth_potentials_buffer::RosenbluthPotentialData,
                fixed_background_plasma::Nothing,
                Zs::Float64,ms::Float64,c0refs::Float64,u0refs::Float64,
                vpa::FiniteElementCoordinate,vperp::FiniteElementCoordinate)
    # do nothing
    return nothing
end
function sum_cross_species_rosenbluth_potentials!(
                rosenbluth_potentials::RosenbluthPotentialData,
                rosenbluth_potentials_buffer::RosenbluthPotentialData,
                rosenbluth_potentials_s::Nothing,
                species::Nothing,
                Zs::Float64,ms::Float64,c0refs::Float64,u0refs::Float64,
                vpa::FiniteElementCoordinate,vperp::FiniteElementCoordinate)
    # do nothing
    return nothing
end
function sum_cross_species_rosenbluth_potentials!(
                rosenbluth_potentials_total::RosenbluthPotentialData,
                rosenbluth_potentials_buffer::RosenbluthPotentialData,
                rosenbluth_potentials_s::Vector{RosenbluthPotentialData},
                species::SpeciesData,
                Zs::Float64,ms::Float64,c0refs::Float64,u0refs::Float64,
                vpa::FiniteElementCoordinate,vperp::FiniteElementCoordinate)
    dHdvpa = rosenbluth_potentials_total.dHdvpa
    dHdvperp = rosenbluth_potentials_total.dHdvperp
    d2Gdvperp2 = rosenbluth_potentials_total.d2Gdvperp2
    d2Gdvpa2 = rosenbluth_potentials_total.d2Gdvpa2
    d2Gdvperpdvpa = rosenbluth_potentials_total.d2Gdvperpdvpa
    mass = species.mass
    zeds = species.zeds
    c0ref = species.c0ref
    u0ref = species.u0ref
    n0ref = species.n0ref
    # buffer arrays containing interpolated rosenbluth potentials
    rp = rosenbluth_potentials_buffer
    for isp in 1:species.n
        # struct for Rosenbluth potentials for species s'
        Zsp = zeds[isp]
        msp = mass[isp]
        c0refsp = c0ref[isp]
        n0refsp = n0ref[isp]
        # interpolate/extrapolate rosenbluth potentials onto s grid
        convert_rosenbluth_potentials_from_source_to_other_grid!(
            rp, rosenbluth_potentials_s[isp],
            vpa, vperp, c0refsp, u0ref[isp], c0refs, u0refs)
        # add the contribution from species s' to the total
        # note that Coulomb logarithm factors are missing
        G_factor = ((Zs*Zsp/ms)^2)*(n0refsp/((c0refs^2)*c0refsp))
        H_factor = (((Zs*Zsp)^2)/(ms*msp))*(n0refsp/(c0refs*(c0refsp^2)))
        @. d2Gdvpa2 += rp.d2Gdvpa2*G_factor
        @. d2Gdvperpdvpa += rp.d2Gdvperpdvpa*G_factor
        @. d2Gdvperp2 += rp.d2Gdvperp2*G_factor
        @. dHdvpa += rp.dHdvpa*H_factor
        @. dHdvperp += rp.dHdvperp*H_factor
    end
    return nothing
end

"""
Function to interpolate `f(vpa,vperp)` from one
velocity grid to another, assuming that both
grids are represented by `(vpa,vperp)` in normalised units,
but have different normalisation factors
defining the meaning of these grids in physical units. E.g.,

     vpai, vperpi = ci * vpa, ci * vperp
     vpae, vperpe = ce * vpa, ce * vperp

with `ci = sqrt(Ti/mi)`, `ce = sqrt(Te/mi)`

`scalefac = ci / ce` is the ratio of the
two reference speeds.
"""
function interpolate_2D_vspace!(pdf_out::Tpdf, pdf_in::Tpdf,
            vpa::FiniteElementCoordinate,
            vperp::FiniteElementCoordinate,
            scalefac::Float64) where Tpdf <: AbstractArray{Float64,2}

    # loop over points in the output interpolated dataset
    @inbounds for ivperp in 1:vperp.n
        vperp_val = vperp.grid[ivperp]*scalefac
        # get element for interpolation data
        if !value_in_coordinate_domain(vperp_val,vperp) # vperp_interp outside of range of vperp.grid
            @inbounds for ivpa in 1:vpa.n
                pdf_out[ivpa,ivperp] = 0.0
            end
            continue
        else
            # get data for interpolation
            iel_vperp = get_ielement(vperp_val,vperp)
            vperp_lpoly_data = vperp.lpoly_data[iel_vperp]
            vperp_igrid_full = @view vperp.igrid_full[:,iel_vperp]
        end
        @inbounds for ivpa in 1:vpa.n
            vpa_val = vpa.grid[ivpa]*scalefac
            # get element for interpolation data
            if !value_in_coordinate_domain(vpa_val,vpa) # vpa_interp outside of range of vpa.grid
                pdf_out[ivpa,ivperp] = 0.0
                continue
            else
                iel_vpa = get_ielement(vpa_val,vpa)
                # get data for interpolation
                vpa_lpoly_data = vpa.lpoly_data[iel_vpa]
                vpa_igrid_full = @view vpa.igrid_full[:,iel_vpa]
                # do the interpolation
                pdf_out[ivpa,ivperp] = interpolate_2D(vpa_lpoly_data,vpa_igrid_full,vpa.ngrid,vpa_val,
                                        vperp_lpoly_data,vperp_igrid_full,vperp.ngrid,vperp_val,pdf_in)
            end
        end
    end
    return nothing
end

function interpolate_2D(vpa_lpoly_data::LagrangePolyData,vpa_igrid_full::Tvector,vpa_ngrid::Int64,vpa_val::Float64,
            vperp_lpoly_data::LagrangePolyData,vperp_igrid_full::Tvector,vperp_ngrid::Int64,vperp_val::Float64,
            pdf_in::Tpdf) where {Tvector <: AbstractArray{Int64,1}, Tpdf <: AbstractArray{Float64,2}}
    result = 0.0
    for ivperpgrid in 1:vperp_ngrid
        # index for referencing pdf_in on orginal grid
        ivperpp = vperp_igrid_full[ivperpgrid]
        igrid_vperp_lpoly_data = vperp_lpoly_data.lpoly_data[ivperpgrid]
        # interpolating polynomial value at ivperpp for interpolation
        vperppoly = lagrange_poly(igrid_vperp_lpoly_data,vperp_val)
        for ivpagrid in 1:vpa_ngrid
            # index for referencing pdf_in on orginal grid
            ivpap = vpa_igrid_full[ivpagrid]
            igrid_vpa_lpoly_data = vpa_lpoly_data.lpoly_data[ivpagrid]
            # interpolating polynomial value at ivpap for interpolation
            vpapoly = lagrange_poly(igrid_vpa_lpoly_data,vpa_val)
            result += vpapoly*vperppoly*pdf_in[ivpap,ivperpp]
        end
    end
    return result
end

"""
Takes a set of Rosenbluth potentials on the grid
where they are sourced (here, species s), and
interpolates them where possible onto the "other" grid
where they are to be used (here species s'= sp).
Where interpolation is not possible, we extrapolate
using the multipole expansion. This method is
consistent with the numerical approximations made by
`calculate_rosenbluth_potentials_via_elliptic_solve!()`
when `boundary_data_option` is either `delta_f_multipole` or
`multipole_expansion`.
"""
function convert_rosenbluth_potentials_from_source_to_other_grid!(
            rosenbluth_potentials_other_grid::RosenbluthPotentialData,
            rosenbluth_potentials_source_grid::RosenbluthPotentialData,
            vpa::FiniteElementCoordinate, vperp::FiniteElementCoordinate,
            c0ref_source::Float64, u0ref_source::Float64,
            c0ref_other::Float64, u0ref_other::Float64;
            calculate_GG=false::Bool,calculate_dGdvperp=false::Bool,
            calculate_HH=false::Bool,test_identity_conversion=false::Bool)
    # denote source grid with s, other grid with s' = sp
    c0refs = c0ref_source
    u0refs = u0ref_source
    c0refsp = c0ref_other
    u0refsp = u0ref_other
    zero = 1.0e-13
    if !test_identity_conversion && (abs(c0refs - c0refsp) < zero) && (abs(u0refs - u0refsp) < zero)
        # reference velocities are identical, copy data directly without interpolation
        if calculate_GG
            @. rosenbluth_potentials_other_grid.GG = rosenbluth_potentials_source_grid.GG
        end
        if calculate_dGdvperp
            @. rosenbluth_potentials_other_grid.dGdvperp = rosenbluth_potentials_source_grid.dGdvperp
        end
        if calculate_HH
            @. rosenbluth_potentials_other_grid.HH = rosenbluth_potentials_source_grid.HH
        end
        @. rosenbluth_potentials_other_grid.dHdvpa = rosenbluth_potentials_source_grid.dHdvpa
        @. rosenbluth_potentials_other_grid.dHdvperp = rosenbluth_potentials_source_grid.dHdvperp
        @. rosenbluth_potentials_other_grid.d2Gdvperp2 = rosenbluth_potentials_source_grid.d2Gdvperp2
        @. rosenbluth_potentials_other_grid.d2Gdvperpdvpa = rosenbluth_potentials_source_grid.d2Gdvperpdvpa
        @. rosenbluth_potentials_other_grid.d2Gdvpa2 = rosenbluth_potentials_source_grid.d2Gdvpa2
    else # unlike reference velocities, interpolate+multipole expand
        # # get index limits on primed species vpa, vperp grids for interpolation
        # # maximum value of vperp on s' grid that can be interpolated
        # vperp_max_sp = (c0refs/c0refsp)*vperp.grid[end]
        # # maximum and minimum values of vpa on s' grid that can be interpolated
        # vpa_max_sp = (c0refs/c0refsp)*vpa.grid[end] + (u0refs - u0refsp)/c0refsp
        # vpa_min_sp = (c0refs/c0refsp)*vpa.grid[1] + (u0refs - u0refsp)/c0refsp
        # ivperp_max_sp = igrid_lookup(vperp_max_sp, vperp, vperp.n, 0)
        # ivpa_max_sp = igrid_lookup(vpa_max_sp, vpa, vpa.n, 0)
        # ivpa_min_sp = igrid_lookup(vpa_min_sp, vpa, 1, 1)
        ivperp_max_sp, ivpa_min_sp, ivpa_max_sp = vpa_vperp_interpolation_limits(c0refs,u0refs,
                                                        c0refsp,u0refsp,vpa,vperp)
        # println("ivperp_max_sp=$ivperp_max_sp")
        # println("ivpa_max_sp=$ivpa_max_sp")
        # println("ivpa_min_sp=$ivpa_min_sp")
        # get the moments of F used for the multipole expansion on the unprimed (source species) grid
        expansion_data = rosenbluth_potentials_source_grid.multipole_expansion_moments
        # use interpolation and extrapolation from the multipole expansion
        if calculate_GG
            rosenbluth_potential_to_other_grid!(GLabel(), rosenbluth_potentials_other_grid.GG,
                rosenbluth_potentials_source_grid.GG, expansion_data,
                vpa, vperp, c0refs, u0refs, c0refsp, u0refsp,
                ivperp_max_sp,ivpa_min_sp,ivpa_max_sp)
        end
        if calculate_dGdvperp
            rosenbluth_potential_to_other_grid!(dGdvperpLabel(), rosenbluth_potentials_other_grid.dGdvperp,
                rosenbluth_potentials_source_grid.dGdvperp, expansion_data,
                vpa, vperp, c0refs, u0refs, c0refsp, u0refsp,
                ivperp_max_sp,ivpa_min_sp,ivpa_max_sp)
        end
        if calculate_HH
            rosenbluth_potential_to_other_grid!(HLabel(), rosenbluth_potentials_other_grid.HH,
                rosenbluth_potentials_source_grid.HH, expansion_data,
                vpa, vperp, c0refs, u0refs, c0refsp, u0refsp,
                ivperp_max_sp,ivpa_min_sp,ivpa_max_sp)
        end
        rosenbluth_potential_to_other_grid!(dHdvpaLabel(), rosenbluth_potentials_other_grid.dHdvpa,
            rosenbluth_potentials_source_grid.dHdvpa, expansion_data,
            vpa, vperp, c0refs, u0refs, c0refsp, u0refsp,
            ivperp_max_sp,ivpa_min_sp,ivpa_max_sp)
        rosenbluth_potential_to_other_grid!(dHdvperpLabel(), rosenbluth_potentials_other_grid.dHdvperp,
            rosenbluth_potentials_source_grid.dHdvperp, expansion_data,
            vpa, vperp, c0refs, u0refs, c0refsp, u0refsp,
            ivperp_max_sp,ivpa_min_sp,ivpa_max_sp)
        rosenbluth_potential_to_other_grid!(d2Gdvperp2Label(), rosenbluth_potentials_other_grid.d2Gdvperp2,
            rosenbluth_potentials_source_grid.d2Gdvperp2, expansion_data,
            vpa, vperp, c0refs, u0refs, c0refsp, u0refsp,
            ivperp_max_sp,ivpa_min_sp,ivpa_max_sp)
        rosenbluth_potential_to_other_grid!(d2GdvperpdvpaLabel(), rosenbluth_potentials_other_grid.d2Gdvperpdvpa,
            rosenbluth_potentials_source_grid.d2Gdvperpdvpa, expansion_data,
            vpa, vperp, c0refs, u0refs, c0refsp, u0refsp,
            ivperp_max_sp,ivpa_min_sp,ivpa_max_sp)
        rosenbluth_potential_to_other_grid!(d2Gdvpa2Label(), rosenbluth_potentials_other_grid.d2Gdvpa2,
            rosenbluth_potentials_source_grid.d2Gdvpa2, expansion_data,
            vpa, vperp, c0refs, u0refs, c0refsp, u0refsp,
            ivperp_max_sp,ivpa_min_sp,ivpa_max_sp)
    end
    return nothing
end

function vpa_vperp_interpolation_limits(
            c0refs::Float64, u0refs::Float64,
            c0refsp::Float64, u0refsp::Float64,
            vpa::FiniteElementCoordinate,
            vperp::FiniteElementCoordinate)
    # get index limits on primed species vpa, vperp grids for interpolation
    # maximum value of vperp on s' grid that can be interpolated
    vperp_max_sp = (c0refs/c0refsp)*vperp.grid[end]
    # maximum and minimum values of vpa on s' grid that can be interpolated
    vpa_max_sp = (c0refs/c0refsp)*vpa.grid[end] + (u0refs - u0refsp)/c0refsp
    vpa_min_sp = (c0refs/c0refsp)*vpa.grid[1] + (u0refs - u0refsp)/c0refsp
    ivperp_max_sp = igrid_lookup(vperp_max_sp, vperp, vperp.n, 0)
    ivpa_max_sp = igrid_lookup(vpa_max_sp, vpa, vpa.n, 0)
    ivpa_min_sp = igrid_lookup(vpa_min_sp, vpa, 1, 1)
    return ivperp_max_sp, ivpa_min_sp, ivpa_max_sp
end
function vpa_s(vpa_sp::Float64,c0refs::Float64,u0refs::Float64,
            c0refsp::Float64,u0refsp::Float64)
    return (c0refsp*vpa_sp + (u0refsp - u0refs))/c0refs
end
function vperp_s(vperp_sp::Float64,c0refs::Float64,c0refsp::Float64)
    return c0refsp*vperp_sp/c0refs
end
function rosenbluth_potential_to_other_grid!(label::Tlabel,
    rosenbluth_potential_other_grid::Tpdf,
    rosenbluth_potential_source_grid::Tpdf,
    expansion_data::Union{Vector{Float64},DeltaFMultipoleMoments},
    vpa::FiniteElementCoordinate,vperp::FiniteElementCoordinate,
    c0ref_source::Float64, u0ref_source::Float64,
    c0ref_other::Float64, u0ref_other::Float64,
    # max and min indices to interpolate, on the other (s') grid
    ivperp_max::Int64,ivpa_min::Int64,ivpa_max::Int64
    ) where {Tlabel <: AbstractRosenbluthPotentialLabel, Tpdf <: AbstractArray{Float64,2}}
    # denote source grid with s, other grid with s' = sp
    c0refs = c0ref_source
    u0refs = u0ref_source
    c0refsp = c0ref_other
    u0refsp = u0ref_other
    # loop over the different regions of the s' grid
    for ivperp in 1:ivperp_max
        vperp_s_val = vperp_s(vperp.grid[ivperp],c0refs,c0refsp)
        for ivpa in 1:ivpa_min-1
            # multipole expand below the minimum ivpa in the s' grid
            vpa_s_val = vpa_s(vpa.grid[ivpa],c0refs,u0refs,c0refsp,u0refsp)
            rosenbluth_potential_other_grid[ivpa,ivperp] = multipole_series(label,vpa_s_val,vperp_s_val,expansion_data)
        end
        # get vperp element for interpolation data
        iel_vperp = get_ielement(vperp_s_val,vperp)
        # get data for interpolation
        vperp_lpoly_data = vperp.lpoly_data[iel_vperp]
        vperp_igrid_full = @view vperp.igrid_full[:,iel_vperp]
        for ivpa in ivpa_min:ivpa_max
            vpa_s_val = vpa_s(vpa.grid[ivpa],c0refs,u0refs,c0refsp,u0refsp)
            # get vpa element for interpolation data
            iel_vpa = get_ielement(vpa_s_val,vpa)
            # get data for interpolation
            vpa_lpoly_data = vpa.lpoly_data[iel_vpa]
            vpa_igrid_full = @view vpa.igrid_full[:,iel_vpa]
            # interpolate between ivpa min and max in s' grid
            rosenbluth_potential_other_grid[ivpa,ivperp] = interpolate_2D(vpa_lpoly_data,vpa_igrid_full,vpa.ngrid,vpa_s_val,
                                        vperp_lpoly_data,vperp_igrid_full,vperp.ngrid,vperp_s_val,rosenbluth_potential_source_grid)
        end
        for ivpa in ivpa_max+1:vpa.n
            # multipole expand above the maximum ivpa in the s' grid
            vpa_s_val = vpa_s(vpa.grid[ivpa],c0refs,u0refs,c0refsp,u0refsp)
            rosenbluth_potential_other_grid[ivpa,ivperp] = multipole_series(label,vpa_s_val,vperp_s_val,expansion_data)
        end
    end
    for ivperp in ivperp_max+1:vperp.n
        vperp_s_val = vperp_s(vperp.grid[ivperp],c0refs,c0refsp)
        for ivpa in 1:vpa.n
            # multipole expand everywhere above the maximum ivperp in the s' grid
            vpa_s_val = vpa_s(vpa.grid[ivpa],c0refs,u0refs,c0refsp,u0refsp)
            rosenbluth_potential_other_grid[ivpa,ivperp] = multipole_series(label,vpa_s_val,vperp_s_val,expansion_data)
        end
    end
    return nothing
end

"""
Function to find the nearest index corresponding to a coordinate value
 v - value of coord to use in root find
 coord - coordinate struct
 ilim_default - integer value to return if root find fails
 p - integer to shift index when v falls between points
"""
function igrid_lookup(v::Float64, coord::FiniteElementCoordinate, ilim_default::Int64, p::Int64)
    zero = 1.0e-14
    if v < coord.grid[1]
        # v lower than lowest grid point
        return p
    end
    if v > coord.grid[coord.n]
        # v larger than largest grid point
        return coord.n + p
    end
    ilim = ilim_default
    for i in 1:coord.n-1
        x1 = v - coord.grid[i]
        x2 = coord.grid[i+1] - v
        # check for intermediate crossings
        if x1*x2 > zero
            ilim = i+p
            break
        # check for grid points
        elseif abs(x1) < 100*zero
            ilim = i
            break
        end
    end
    # check final grid point
    if abs(coord.grid[coord.n] - v) < 100*zero
        ilim = coord.n
    end
    return ilim
end

"""
Function that solves `A x = b` for a matrix of the form
```math
\\begin{array}{ccc}
A_{00} & 0 & A_{02} \\\\
0 & A_{11} & A_{12} \\\\
A_{02} & A_{12} & A_{22} \\\\
\\end{array}
```
appropriate for the moment numerical conserving terms used in
the Fokker-Planck collision operator.
"""
function symmetric_matrix_inverse(A00::Float64,A02::Float64,
                    A11::Float64,A12::Float64,A22::Float64,
                    b0::Float64,b1::Float64,b2::Float64)
    # matrix determinant
    detA = A00*(A11*A22 - A12^2) - A11*A02^2
    # cofactors C (also a symmetric matrix)
    C00 = A11*A22 - A12^2
    C01 = A12*A02
    C02 = -A11*A02
    C11 = A00*A22 - A02^2
    C12 = -A00*A12
    C22 = A00*A11
    x0 = ( C00*b0 + C01*b1 + C02*b2 )/detA
    x1 = ( C01*b0 + C11*b1 + C12*b2 )/detA
    x2 = ( C02*b0 + C12*b1 + C22*b2 )/detA
    #println("b0: ",b0," b1: ",b1," b2: ",b2)
    #println("A00: ",A00," A02: ",A02," A11: ",A11," A12: ",A12," A22: ",A22, " detA: ",detA)
    #println("C00: ",C00," C02: ",C02," C11: ",C11," C12: ",C12," C22: ",C22)
    #println("x0: ",x0," x1: ",x1," x2: ",x2)
    return x0, x1, x2
end

"""
Function that solves `A x = b` for a matrix of the form
```math
\\begin{array}{ccc}
A_{00} & A_{01} & A_{02} \\\\
A_{01} & A_{11} & A_{12} \\\\
A_{02} & A_{12} & A_{22} \\\\
\\end{array}
```
appropriate for moment numerical conserving terms.

"""
function symmetric_matrix_inverse(A00::Float64,A01::Float64,A02::Float64,
                            A11::Float64,A12::Float64,A22::Float64,
                            b0::Float64,b1::Float64,b2::Float64)
    # matrix determinant
    detA = A00*(A11*A22 - A12^2) - A01*(A01*A22 - A12*A02) + A02*(A01*A12 - A11*A02)
    # cofactors C (also a symmetric matrix)
    C00 = A11*A22 - A12^2
    C01 = A12*A02 - A01*A22
    C02 = A01*A12 -A11*A02
    C11 = A00*A22 - A02^2
    C12 = A01*A02 -A00*A12
    C22 = A00*A11 - A01^2
    x0 = ( C00*b0 + C01*b1 + C02*b2 )/detA
    x1 = ( C01*b0 + C11*b1 + C12*b2 )/detA
    x2 = ( C02*b0 + C12*b1 + C22*b2 )/detA
    #println("b0: ",b0," b1: ",b1," b2: ",b2)
    #println("A00: ",A00," A02: ",A02," A11: ",A11," A12: ",A12," A22: ",A22, " detA: ",detA)
    #println("C00: ",C00," C02: ",C02," C11: ",C11," C12: ",C12," C22: ",C22)
    #println("x0: ",x0," x1: ",x1," x2: ",x2)
    return x0, x1, x2
end

"""
Function that solves `A x = b` for a nearly-symmetric matrix of the form
```math
\\begin{array}{cccc}
A_{00} & 0      & 0      & A_{03} \\\\
0      & A_{11} & 0      & A_{13} \\\\
0      & 0      & A_{22} & A_{23} \\\\
A_{03} & A_{13} & A_{32} & A_{33}\\\\
\\end{array}
```
appropriate for cross-species numerical conserving terms.

"""
function matrix_inverse(A00::Float64,A03::Float64,A11::Float64,A13::Float64,
                            A22::Float64,A23::Float64,A32::Float64,A33::Float64,
                            b0::Float64,b1::Float64,b2::Float64,b3::Float64)
    # matrix determinant
    detA = (A00*A11*A22*A33 - A00*A11*A23*A32 - A00*A13^2*A22 - A03^2*A11*A22)
    # solve A x = b
    x0 = ( -A03*A11*A22*b3 + A03*A11*A32*b2 + A03*A13*A22*b1 + A11*A22*A33*b0 - A11*A23*A32*b0 - A13^2*A22*b0)/detA
    x1 = ( -A00*A13*A22*b3 + A00*A13*A32*b2 + A00*A22*A33*b1 - A00*A23*A32*b1 - A03^2*A22*b1 + A03*A13*A22*b0)/detA
    x2 = ( -A00*A11*A23*b3 + A00*A11*A33*b2 - A00*A13^2*b2 + A00*A13*A23*b1 - A03^2*A11*b2 + A03*A11*A23*b0 )/detA
    x3 = (  A00*A11*A22*b3 - A00*A11*A32*b2 - A00*A13*A22*b1 - A03*A11*A22*b0 )/ detA
    return x0, x1, x2, x3
end
function matrix_inverse(A00::Float64,A01::Float64,A10::Float64,A11::Float64,
                        b0::Float64,b1::Float64)
    # matrix determinant
    detA = A00*A11 - A01*A10
    # solve A x = b
    x0 = (b0*A11 - b1*A01)/detA
    x1 = (-b0*A10 + b1*A00)/detA
    return x0, x1
end

"""
Function that applies numerical-error correcting terms to ensure
numerical conservation of the moments `density, upar, pressure` in the self-collision operator.
Modifies the collision operator such that the operator becomes
```math
C_{ss} = C^\\ast_{ss}[F_s,F_{s}] - \\left(x_0 + x_1(v_{\\|}-u_{\\|})+ x_2(v_\\perp^2 +(v_{\\|}-u_{\\|})^2)\\right)F_s
```
where \$C^\\ast_{ss}[F_s,F_{s}]\$ is the weak-form self-collision operator computed using
the finite-element implementation, \$u_{\\|}\$ is the parallel velocity of \$F_s\$,
and \$x_0,x_1,x_2\$ are parameters that are chosen so that \$C_{ss}\$
conserves density, parallel velocity and pressure of \$F_s\$.
"""
# corrections to preserve the symmetry of the collision operators
function conserving_corrections!(CC::Tpdf1, pdf_in::Tpdf2, nuref::Float64,
            fkpl_arrays::FokkerPlanckWeakformArrays) where {Tpdf1 <: AbstractArray{Float64,3}, Tpdf2 <: AbstractArray{Float64,3}}
    @inbounds begin
        if fkpl_arrays.multi_species_operator_option == single_assembly_per_species
            # calculate necessary moments and store in fkpl_arrays
            calculate_collision_moments!(pdf_in,nuref,fkpl_arrays)
        end
        # extract precomputed variables
        vpa = fkpl_arrays.vpa
        vperp = fkpl_arrays.vperp
        species = fkpl_arrays.species
        mass = species.mass
        c0ref = species.c0ref
        u0ref = species.u0ref
        n0ref = species.n0ref
        # moments of collisions for each cross-species pair
        delta_n_sp_s = fkpl_arrays.delta_n_sp_s
        delta_m_sp_s = fkpl_arrays.delta_m_sp_s
        delta_p_sp_s = fkpl_arrays.delta_p_sp_s
        # moments of the pdf for each species
        density = fkpl_arrays.density
        upar = fkpl_arrays.upar
        pressure = fkpl_arrays.pressure
        ppar = fkpl_arrays.ppar
        qpar = fkpl_arrays.qpar
        rmom = fkpl_arrays.rmom
        # collect the calculated moments
        for is in 1:species.n
            @views density[is] = n0ref[is]*get_density(pdf_in[:,:,is], vpa, vperp)
            @views upar[is] = c0ref[is]*get_upar(pdf_in[:,:,is], vpa, vperp, density[is]/n0ref[is]) + u0ref[is]
            @views pressure[is] = n0ref[is]*(c0ref[is]^2)*get_pressure(pdf_in[:,:,is], vpa, vperp, (upar[is]-u0ref[is])/c0ref[is], mass[is])
            @views ppar[is] = n0ref[is]*(c0ref[is]^2)*get_ppar(pdf_in[:,:,is], vpa, vperp, (upar[is]-u0ref[is])/c0ref[is], mass[is])
            @views qpar[is] = n0ref[is]*(c0ref[is]^3)*get_qpar(pdf_in[:,:,is], vpa, vperp, (upar[is]-u0ref[is])/c0ref[is], mass[is])
            @views rmom[is] = n0ref[is]*(c0ref[is]^4)*get_rmom(pdf_in[:,:,is], vpa, vperp, (upar[is]-u0ref[is])/c0ref[is], mass[is])
        end
        # correction coefficients
        zcoeffs = fkpl_arrays.correction_coeffs_z
        # first get the correction coefficients
        for is in 1:species.n
            # self collision terms
            # form the appropriate matrix coefficients
            b0, b1, b2 = mass[is]*delta_n_sp_s[is,is], delta_m_sp_s[is,is], 3.0*delta_p_sp_s[is,is]
            A00, A02, A11, A12, A22 = mass[is]*density[is], 3.0*pressure[is], ppar[is], 2.0*qpar[is], rmom[is]
            # obtain the coefficients for the corrections
            (x0, x1, x2) = symmetric_matrix_inverse(A00,A02,A11,A12,A22,b0,b1,b2)
            zcoeffs[1,is,is] = x0
            zcoeffs[2,is,is] = x1
            zcoeffs[3,is,is] = x2
            # cross species terms
            for isp in is+1:species.n
                A00 = mass[is]*density[is]
                A03 = 3.0*pressure[is]
                A11 = mass[isp]*density[isp]
                A13 = 3.0*pressure[isp]
                A22 = ppar[is] + ppar[isp]
                A23 = 2*(qpar[is]+qpar[isp])
                A32 = A23 + (ppar[is] - ppar[isp])*(upar[is] - upar[isp])
                A33 = rmom[is] + rmom[isp] + 2.0*(qpar[is] - qpar[isp])*(upar[is] - upar[isp])
                b0 = mass[is]*delta_n_sp_s[isp,is]
                b1 = mass[isp]*delta_n_sp_s[is,isp]
                b2 = delta_m_sp_s[isp,is] + delta_m_sp_s[is,isp]
                b3 = (3.0*(delta_p_sp_s[isp,is] + delta_p_sp_s[is,isp]) +
                    (upar[is] - upar[isp])*(delta_m_sp_s[isp,is] - delta_m_sp_s[is,isp]))
                # obtain the coefficients for the corrections
                (x0, x1, x2, x3) = matrix_inverse(A00,A03,A11,A13,
                                                    A22,A23,A32,A33,
                                                    b0,b1,b2,b3)
                # corrections for species s
                zcoeffs[1,isp,is] = x0
                zcoeffs[2,isp,is] = x2
                zcoeffs[3,isp,is] = x3
                # corrections for species s'
                zcoeffs[1,is,isp] = x1
                zcoeffs[2,is,isp] = x2
                zcoeffs[3,is,isp] = x3
            end
        end
        # correct CC
        for is in 1:species.n
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    wpar = vpa.grid[ivpa] - (upar[is] - u0ref[is])/c0ref[is]
                    for isp in 1:species.n
                        x0 = zcoeffs[1,isp,is]
                        x1 = zcoeffs[2,isp,is]*c0ref[is]
                        x2 = zcoeffs[3,isp,is]*c0ref[is]^2
                        CC[ivpa,ivperp,is] -= (x0 + x1*wpar + x2*(vperp.grid[ivperp]^2 + wpar^2) )*pdf_in[ivpa,ivperp,is]
                    end
                end
            end
        end
    end # @inbounds
    return nothing
end
# corrections to preserve the density, total momentum and total energy in the pdf(vpa,vperp,species)
# only possible to apply to closed systems without fixed species or sources and sinks
function conserving_corrections!(pdf_new::Tpdf1, pdf_old::Tpdf2,
            fkpl_arrays::FokkerPlanckWeakformArrays,
            source_data::Nothing) where {Tpdf1 <: AbstractArray{Float64,3}, Tpdf2 <: AbstractArray{Float64,3}}
    return conserving_corrections!(pdf_new, pdf_old, fkpl_arrays,
        fkpl_arrays.fixed_background_plasma, source_data)
end
function conserving_corrections!(pdf_new::Tpdf1, pdf_old::Tpdf2,
            fkpl_arrays::FokkerPlanckWeakformArrays,
            source_data::SlowingDownSourceData) where {Tpdf1 <: AbstractArray{Float64,3}, Tpdf2 <: AbstractArray{Float64,3}}
    # do nothing
    return nothing
end
function conserving_corrections!(pdf_new::Tpdf1, pdf_old::Tpdf2,
            fkpl_arrays::FokkerPlanckWeakformArrays,
            fixed_background_plasma::FixedBackgroundPlasmaData,
            source_data::Union{Nothing,SlowingDownSourceData}) where {Tpdf1 <: AbstractArray{Float64,3}, Tpdf2 <: AbstractArray{Float64,3}}
    # do nothing
    return nothing
end
# only if there are no sources and no fixed background can we apply these corrections to F
function conserving_corrections!(pdf_new::Tpdf1, pdf_old::Tpdf2,
            fkpl_arrays::FokkerPlanckWeakformArrays,
            fixed_background_plasma::Nothing,
            source_data::Nothing) where {Tpdf1 <: AbstractArray{Float64,3}, Tpdf2 <: AbstractArray{Float64,3}}
    vpa = fkpl_arrays.vpa
    vperp = fkpl_arrays.vperp
    species = fkpl_arrays.species
    mass = species.mass
    c0ref = species.c0ref
    u0ref = species.u0ref
    n0ref = species.n0ref
    # moments of the pdf for each species
    density = fkpl_arrays.density
    upar = fkpl_arrays.upar
    pressure = fkpl_arrays.pressure
    ppar = fkpl_arrays.ppar
    qpar = fkpl_arrays.qpar
    rmom = fkpl_arrays.rmom
    delta_n = fkpl_arrays.delta_n
    delta_P = fkpl_arrays.delta_P
    delta_E = fkpl_arrays.delta_E

    # compute deltaF = F* - F^n, where F* is the Fnew from the uncorrected FP solve
    delta_pdf = fkpl_arrays.delta_pdf
    @inbounds begin
        for is in 1:species.n
            @views enforce_vpavperp_BCs!(pdf_new[:,:,is],vpa,vperp)
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    delta_pdf[ivpa,ivperp,is] = pdf_new[ivpa,ivperp,is] - pdf_old[ivpa,ivperp,is]
                end
            end
        end

        for is in 1:species.n
            # compute moments of the input pdf_new
            @views density[is] = n0ref[is]*get_density(pdf_new[:,:,is], vpa, vperp)
            @views up0 = get_upar(pdf_new[:,:,is], vpa, vperp, density[is]/n0ref[is]) # upar in reference frame of species s
            upar[is] = c0ref[is]*up0 + u0ref[is]
            @views pressure[is] = n0ref[is]*(c0ref[is]^2)*get_pressure(pdf_new[:,:,is], vpa, vperp, up0, mass[is])
            @views ppar[is] = n0ref[is]*(c0ref[is]^2)*get_ppar(pdf_new[:,:,is], vpa, vperp, up0, mass[is])
            @views qpar[is] = n0ref[is]*(c0ref[is]^3)*get_qpar(pdf_new[:,:,is], vpa, vperp, up0, mass[is])
            @views rmom[is] = n0ref[is]*(c0ref[is]^4)*get_rmom(pdf_new[:,:,is], vpa, vperp, up0, mass[is])
            # compute necessary moments of delta_pdf
            @views delta_n[is] = n0ref[is]*get_density(delta_pdf[:,:,is], vpa, vperp)
            @views delta_P[is] = mass[is]*(n0ref[is]*c0ref[is]*get_upar(delta_pdf[:,:,is], vpa, vperp, 1.0)
                                                        + delta_n[is]*u0ref[is])
            upzero = (0.0 - u0ref[is])/c0ref[is] # vpa in species s frame corresponding to vpa = 0 in lab frame
            @views delta_E[is] = 1.5*n0ref[is]*(c0ref[is]^2)*get_pressure(delta_pdf[:,:,is], vpa, vperp, upzero, mass[is])
        end

        b0, b1 = 0.0, 0.0
        A00, A01, A10, A11 =  0.0, 0.0, 0.0, 0.0
        for is in 1:species.n
            b0 += delta_P[is] - mass[is]*upar[is]*delta_n[is]
            b1 += 2.0*delta_E[is] - delta_n[is]*(mass[is]*upar[is]^2 + 3.0*pressure[is]/density[is])
            A00 += ppar[is]
            A01 += 2.0*qpar[is]
            A10 += 2.0*(qpar[is] + ppar[is]*upar[is])
            A11 += rmom[is] + 4.0*upar[is]*qpar[is] - 9.0*(pressure[is]^2)/(mass[is]*density[is])
        end

        # obtain the coefficients for the corrections
        (x1, x2) = matrix_inverse(A00,A01,A10,A11,b0,b1)

        # correct pdf_new with polynomial correction * pdf_new
        for is in 1:species.n
            x0 = (delta_n[is]/density[is]) - 3.0*(pressure[is]/(mass[is]*density[is]))*x2
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    wpar = vpa.grid[ivpa] - (upar[is] - u0ref[is])/c0ref[is]
                    pdf_new[ivpa,ivperp,is] -= (x0 + x1*c0ref[is]*wpar + x2*(c0ref[is]^2)*(vperp.grid[ivperp]^2 + wpar^2) )*pdf_new[ivpa,ivperp,is]
                end
            end
        end
    end
    return nothing
end

"""
Function that applies a numerical-error correcting term to ensure
numerical conservation of the `density` in the collision operator.
```math
C_{ss^\\prime} = C^\\ast_{ss}[F_s,F_{s^\\prime}] - x_0 F_s
```
where \$C^\\ast_{ss}[F_s,F_{s^\\prime}]\$ is the weak-form collision operator computed using
the finite-element implementation.
"""
function density_conserving_correction!(
            CC::Tpdf1, pdf_in::Tpdf2,
            vpa::FiniteElementCoordinate, vperp::FiniteElementCoordinate
            ) where {Tpdf1 <: AbstractArray{Float64,2}, Tpdf2 <: AbstractArray{Float64,2}}
    # compute density of the input pdf
    dens =  get_density(pdf_in, vpa, vperp)

    # compute density of the numerical collision operator
    dn = get_density(CC, vpa, vperp)

    # obtain the coefficient for the correction
    x0 = dn/dens

    # correct CC
    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                CC[ivpa,ivperp] -= x0*pdf_in[ivpa,ivperp]
            end
        end
    end
end
##
# element-wise integration function to get moments of C(vpa,vperp) without assembling C
##
function calculate_collision_moments!(pdf_in::Tpdf,
    nuref::Float64,fkpl_arrays::FokkerPlanckWeakformArrays) where Tpdf <: AbstractArray{Float64,3}
    # call the lower level function after expanding some variables
    vpa = fkpl_arrays.vpa
    vperp = fkpl_arrays.vperp
    species = fkpl_arrays.species
    mass = species.mass
    zeds = species.zeds
    c0ref = species.c0ref
    u0ref = species.u0ref
    n0ref = species.n0ref
    YY_arrays = fkpl_arrays.YY_arrays
    # Rosenbluth potentials for each species
    rosenbluth_potentials_s = fkpl_arrays.rosenbluth_potentials_s
    rosenbluth_potentials = fkpl_arrays.rosenbluth_potentials
    # storage for interpolated rosenbluth potentials
    rp = fkpl_arrays.rosenbluth_potentials_buffer
    # Rosenbluth potentials for passing into function
    d2Gdvperp2 = rosenbluth_potentials.d2Gdvperp2
    d2Gdvpa2 = rosenbluth_potentials.d2Gdvpa2
    d2Gdvperpdvpa = rosenbluth_potentials.d2Gdvperpdvpa
    dHdvperp = rosenbluth_potentials.dHdvperp
    dHdvpa = rosenbluth_potentials.dHdvpa
    # moments of collisions for each cross-species pair
    delta_n_sp_s = fkpl_arrays.delta_n_sp_s
    delta_m_sp_s = fkpl_arrays.delta_m_sp_s
    delta_p_sp_s = fkpl_arrays.delta_p_sp_s
    # moments of the pdf for each species
    density = fkpl_arrays.density
    upar = fkpl_arrays.upar
    # collect the calculated moments needed for the calculation below
    for is in 1:species.n
        @views density[is] = n0ref[is]*get_density(pdf_in[:,:,is], vpa, vperp)
        @views upar[is] = c0ref[is]*get_upar(pdf_in[:,:,is], vpa, vperp, density[is]/n0ref[is]) + u0ref[is]
    end
    # collect the collision integrals
    for is in 1:species.n
        for isp in 1:species.n
            # interpolate/extrapolate rosenbluth potentials onto s grid
            # this should be done once so that work done in
            # sum_cross_species_rosenbluth_potentials!() is not duplicated
            # -- keep current code structure for now
            convert_rosenbluth_potentials_from_source_to_other_grid!(
                rp, rosenbluth_potentials_s[isp],
                vpa, vperp, c0ref[isp], u0ref[isp], c0ref[is], u0ref[is])
            G_factor = ((zeds[is]*zeds[isp]/mass[is])^2)*(n0ref[isp]/((c0ref[is]^2)*c0ref[isp]))
            H_factor = (((zeds[is]*zeds[isp])^2)/(mass[is]*mass[isp]))*(n0ref[isp]/(c0ref[is]*(c0ref[isp]^2)))
            @. d2Gdvperp2 = rp.d2Gdvperp2*G_factor
            @. d2Gdvperpdvpa = rp.d2Gdvperpdvpa*G_factor
            @. d2Gdvpa2 = rp.d2Gdvpa2*G_factor
            @. dHdvpa = rp.dHdvpa*H_factor
            @. dHdvperp = rp.dHdvperp*H_factor
            @views (int_C, int_vpa_C, int_vpa2_C, int_vperp2_C) = integrate_collision_moments(pdf_in[:,:,is],d2Gdvpa2,d2Gdvperpdvpa,
                d2Gdvperp2,dHdvpa,dHdvperp,1.0,1.0,nuref,
                vpa,vperp,YY_arrays)
            up0 = (upar[is] - u0ref[is])/c0ref[is]
            delta_n_sp_s[isp,is] = int_C*n0ref[is]
            delta_m_sp_s[isp,is] = mass[is]*n0ref[is]*c0ref[is]*(int_vpa_C - up0*int_C)
            delta_p_sp_s[isp,is] = (mass[is]*n0ref[is]*(c0ref[is]^2)/3.0)*(
                                    int_vpa2_C - 2*up0*int_vpa_C
                                     + (up0^2)*int_C + int_vperp2_C)
        end
    end
    return nothing
end

function integrate_collision_moments(pdfs::Tpdf1,
    d2Gspdvpa2::Tpdf2,d2Gspdvperpdvpa::Tpdf2,
    d2Gspdvperp2::Tpdf2,dHspdvpa::Tpdf2,dHspdvperp::Tpdf2,
    ms::Float64,msp::Float64,nussp::Float64,
    vpa::FiniteElementCoordinate,
    vperp::FiniteElementCoordinate,
    YY_arrays::CollisionOperatorArrays) where {Tpdf1 <: AbstractArray{Float64,2}, Tpdf2 <: AbstractArray{Float64,2}}
    if vpa.ngrid < 5 || vperp.ngrid < 5
        msg = """ERROR: integrate_collision_moments() is inconsistent with integration weights for vpa.ngrid < 5 or vperp.ngrid < 5
        """
        error(msg)
    end
    if vpa.bc != natural_boundary_condition || vperp.bc != natural_boundary_condition
        msg = """ERROR: integrate_collision_moments() is only consistent with boundary conditions for vpa.bc=natural_boundary_condition and vperp.bc=natural_boundary_condition
        """
        error(msg)
    end
    int_C = 0.0 # by construction
    int_vpa_C = 0.0
    int_vpa2_C = 0.0
    int_vperp2_C = 0.0
    int_C_vec = [int_C, int_vpa_C, int_vpa2_C, int_vperp2_C]
    @inbounds begin
        # assemble integrals of collision operator
        # loop over elements
        for ielement_vperp in 1:vperp.nelement
            MMperp = YY_arrays.MMperp[:,:,ielement_vperp]
            MRperp = YY_arrays.MRperp[:,:,ielement_vperp]
            PPperp = YY_arrays.PPperp[:,:,ielement_vperp]
            PUperp = YY_arrays.PUperp[:,:,ielement_vperp]
            #println("MMperp: ", MMperp)
            #println("PPperp: ",PPperp)
            for ielement_vpa in 1:vpa.nelement
                MMpar = YY_arrays.MMpar[:,:,ielement_vpa]
                MRpar = YY_arrays.MRpar[:,:,ielement_vpa]
                PPpar = YY_arrays.PPpar[:,:,ielement_vpa]
                PUpar = YY_arrays.PUpar[:,:,ielement_vpa]
                #println("MMpar: ", MMpar)
                #println("PPpar: ", PPpar)
                # loop over field positions in each element
                for kvperpp_local in 1:vperp.ngrid
                    kvperpp = vperp.igrid_full[kvperpp_local,ielement_vperp]
                    for kvpap_local in 1:vpa.ngrid
                        kvpap = vpa.igrid_full[kvpap_local,ielement_vpa]
                        d2Gdvpa2kk = d2Gspdvpa2[kvpap,kvperpp]
                        d2Gdvperp2kk = d2Gspdvperp2[kvpap,kvperpp]
                        d2Gdvperpdvpakk = d2Gspdvperpdvpa[kvpap,kvperpp]
                        dHdvpakk = dHspdvpa[kvpap,kvperpp]
                        dHdvperpkk = dHspdvperp[kvpap,kvperpp]
                        for jvperpp_local in 1:vperp.ngrid
                            jvperpp = vperp.igrid_full[jvperpp_local,ielement_vperp]
                            for jvpap_local in 1:vpa.ngrid
                                jvpap = vpa.igrid_full[jvpap_local,ielement_vpa]
                                pdfjj = pdfs[jvpap,jvperpp]
                                # carry out the matrix sum on each 2D element
                                # the three lines represent parallel flux terms
                                # int_vpa_C
                                int_C_vec[2] += -(2.0*pi)*
                                                nussp*pdfjj*(PPpar[jvpap_local,kvpap_local]*MMperp[jvperpp_local,kvperpp_local]*d2Gdvpa2kk +
                                                            MMpar[jvpap_local,kvpap_local]*PPperp[jvperpp_local,kvperpp_local]*d2Gdvperpdvpakk -
                                                            2.0*(ms/msp)*MMpar[jvpap_local,kvpap_local]*MMperp[jvperpp_local,kvperpp_local]*dHdvpakk
                                                            )
                                # the three lines represent parallel flux terms
                                # int_vpa2_C
                                int_C_vec[3] += -(2.0*pi)*
                                                    2.0*nussp*pdfjj*(PUpar[jvpap_local,kvpap_local]*MMperp[jvperpp_local,kvperpp_local]*d2Gdvpa2kk +
                                                                    MRpar[jvpap_local,kvpap_local]*PPperp[jvperpp_local,kvperpp_local]*d2Gdvperpdvpakk -
                                                                    2.0*(ms/msp)*MRpar[jvpap_local,kvpap_local]*MMperp[jvperpp_local,kvperpp_local]*dHdvpakk)
                                # the three lines represent perpendicular flux terms
                                # int_vperp2_C
                                int_C_vec[4] += -(2.0*pi)*
                                                    2.0*nussp*pdfjj*(PPpar[jvpap_local,kvpap_local]*MRperp[jvperpp_local,kvperpp_local]*d2Gdvperpdvpakk +
                                                                    MMpar[jvpap_local,kvpap_local]*PUperp[jvperpp_local,kvperpp_local]*d2Gdvperp2kk -
                                                                    2.0*(ms/msp)*MMpar[jvpap_local,kvpap_local]*MRperp[jvperpp_local,kvperpp_local]*dHdvperpkk)
                            end
                        end
                    end
                end
            end
        end
    end
    return int_C_vec
end

function fokker_planck_collision_operator_solve!(
            CCssp::Tpdf1, ffs_in::Tpdf2,
            rosenbluth_potential_sp_in::RosenbluthPotentialData,
            ms::Float64, msp::Float64, nussp::Float64,
            rhsvpavperp::Array{Float64,2},
            lu_obj_MM::SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64},
            YY_arrays::CollisionOperatorArrays,
            vpa::FiniteElementCoordinate, vperp::FiniteElementCoordinate) where {Tpdf1 <: AbstractArray{Float64,2}, Tpdf2 <: AbstractArray{Float64,2}}
    # assemble the RHS of the collision operator matrix eq
    assemble_explicit_collision_operator_rhs_serial!(rhsvpavperp,ffs_in,
            rosenbluth_potential_sp_in,ms,msp,nussp,vpa,vperp,YY_arrays)
    # solve the collision operator matrix eq
    # sc and rhsc are 1D views of the data in CC and rhsc, created so that we can use
    # the 'matrix solve' functionality of ldiv!() from the LinearAlgebra package
    sc = vec(CCssp)
    rhsc = vec(rhsvpavperp)
    # invert mass matrix and fill fc
    ldiv!(sc, lu_obj_MM, rhsc)
    return nothing
end

end
