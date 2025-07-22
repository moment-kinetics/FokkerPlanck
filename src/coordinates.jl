"""
"""
module coordinates

export first_derivative!,
    finite_element_coordinate,
    scalar_coordinate_inputs,
    # testing
    set_element_boundaries,
    set_element_scale_and_shift

using LinearAlgebra
using SparseArrays: sparse, AbstractSparseArray
using SuiteSparse
using LagrangePolynomials: lagrange_poly_data, lagrange_poly
using FiniteElementMatrices: element_coordinates,
                             lagrange_x,
                             d_lagrange_dx,
                             finite_element_matrix
using FastGaussQuadrature: gausslegendre, gausslobatto, gaussradau
using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float, allocate_int

"""
struct containing information for first derivatives
"""
struct first_derivative_data
    # Local mass matrix = \int phi_i(x) phi_j(x) dx
    MM::Array{mk_float,3}
    # Local first derivative matrix \int phi_i(x) phi'_j(x) dx
    PP::Array{mk_float,3}
    # Local D_ij matrix: defined such that  (d f / dx)_i = \sum_j D_ij f_j
    DD::Array{mk_float,3}
    # dummy array for storing intermediate results
    dummy_rhs::Array{mk_float,1}
    # dummy array for storing solution
    dummy_df::Array{mk_float,1}
    # Assembled 1D mass matrix
    MM1D::AbstractSparseArray{mk_float,mk_int,2}
    # Assembled 1D first derivative matrix
    PP1D::AbstractSparseArray{mk_float,mk_int,2}
    # LU object for mass matrix solve
    lu_MM1D::SuiteSparse.UMFPACK.UmfpackLU{mk_float,mk_int}
    """
    Function to initialise first_derivative_data
    from information from the grid.
    """
    function first_derivative_data(ngrid::mk_int,
                                        nelement::mk_int,
                                        n::mk_int,
                                        igrid_full::Array{mk_int,2},
                                        element_data::Array{element_coordinates,1})
        # Local mass matrix = \int phi_i(x) phi_j(x) dx
        MM = Array{mk_float,3}(undef,ngrid,ngrid,nelement)
        # \int phi_i(x) phi'_j(x) dx
        PP = Array{mk_float,3}(undef,ngrid,ngrid,nelement)
        # D_ij is the matrix such that  (d f / dx)_i = \sum_j D_ij f_j
        DD = Array{mk_float,3}(undef,ngrid,ngrid,nelement)
        # get local matrices on each element
        for ielement in 1:nelement
            xdata = element_data[ielement]
            @views MM[:,:,ielement] = finite_element_matrix(lagrange_x,lagrange_x,0,xdata)
            @views PP[:,:,ielement] = finite_element_matrix(lagrange_x,d_lagrange_dx,0,xdata)
            @views lu_MM_local = lu(MM[:,:,ielement])
            @views ldiv!(DD[:,:,ielement],lu_MM_local,PP[:,:,ielement])
        end
        # dummy arrays
        dummy_rhs = Array{mk_float,1,}(undef,n)
        dummy_df = Array{mk_float,1,}(undef,n)

        # assemble the global matrices with a sparse construction
        MM1D = assemble_1D_operator(MM, ngrid, nelement,
                            n, igrid_full)
        PP1D = assemble_1D_operator(PP, ngrid, nelement,
                            n, igrid_full)
        MM1D_sparse = sparse(MM1D)
        PP1D_sparse = sparse(PP1D)
        lu_MM1D = lu(MM1D_sparse)
        return new(MM,PP,DD,
                dummy_rhs, dummy_df,
                MM1D_sparse,PP1D_sparse,lu_MM1D)
    end
end

"""
Compound index for the sparse assembly of a 1D
finite element matrix. Note that this compound
index is different to the one used internally
by the object created by `SparseArrays.sparse()`.
"""
function icsc1D(ix_local::mk_int,
            ixp_local::mk_int,
            ielement::mk_int,
            ngrid::mk_int,
            nelement::mk_int)
    icsc = 1 + ((ixp_local - 1) + (ix_local - 1)*ngrid +
                (ielement - 1)*(ngrid^2 - 1))
    return icsc
end
function assemble_1D_operator(QQ1D_local::Array{mk_float,3},
                            ngrid::mk_int,
                            nelement::mk_int,
                            n::mk_int,
                            igrid_full::Array{mk_int,2})
    # create the 1D constructor arrays
    nsparse = (nelement - 1)*(ngrid^2 - 1) + ngrid^2
    II = zeros(mk_int,nsparse)
    JJ = zeros(mk_int,nsparse)
    VV = zeros(mk_float,nsparse)
    # loop over elements
    for ielement in 1:nelement
        @views QQ1D = QQ1D_local[:,:,ielement]
        @views igrid = igrid_full[:,ielement]
        # loop over points within elements
        for ixp in 1:ngrid
            for ix in 1:ngrid
                # convert local indices within
                # elements to global indices and
                # the constructor compound index
                ix_global = igrid[ix]
                ixp_global = igrid[ixp]
                isparse = icsc1D(ix,ixp,ielement,ngrid,nelement)
                # assign data
                II[isparse] = ix_global
                JJ[isparse] = ixp_global
                VV[isparse] += QQ1D[ix,ixp]
            end
        end
    end
    # convert constructors to sparse matrix
    QQ1D_global = sparse(II,JJ,VV)
    return QQ1D_global
end

"""
This struct encapsulates the unique information
required to create an instance of
```
    Array{element_coordinates,1}
```
to permit an interface with a single function
for initialising the arrays for the finite element
problem, whilst retaining the convenience of
not having to construct the `element_coordinates`
types manually in all tests.
"""
struct scalar_coordinate_inputs
    # ngrid is number of grid points per element
    ngrid::mk_int
    # nelement is the number of elements in total
    nelement::mk_int
    # Ldomain is the box length in this coordinate
    Ldomain::mk_float
end

"""
enum for boundary conditions imposed
on the evolved solution.
"""
@enum finite_element_boundary_condition_type begin
    # boundary condition imposed by structure of FE matrices
    natural_boundary_condition
    # Zero boundary conditions are imposed on the FE matrices and the solution
    zero_boundary_condition
end

"""
structure containing basic information related to coordinates
"""
struct finite_element_coordinate
    # name is the name of the variable associated with this coordiante
    name::String
    # n is the total number of grid points associated with this coordinate
    n::mk_int
    # ngrid is the number of grid points per element in this coordinate
    ngrid::mk_int
    # nelement is the number of elements associated with this coordinate
    nelement::mk_int
    # L is the box length in this coordinate
    L::mk_float
    # grid is the location of the grid points
    grid::Array{mk_float,1}
    # igrid contains the grid point index within the element
    igrid::Array{mk_int,1}
    # ielement contains the element index
    ielement::Array{mk_int,1}
    # imin[j] contains the minimum index on the full grid for element j
    imin::Array{mk_int,1}
    # imax[j] contains the maximum index on the full grid for element j
    imax::Array{mk_int,1}
    # igrid_full[i,j] contains the index of the full grid for the elemental grid point i, on element j
    igrid_full::Array{mk_int,2}
    # bc is the boundary condition option for this coordinate
    bc::finite_element_boundary_condition_type
    # wgts contains the integration weights associated with each grid point
    wgts::Array{mk_float,1}
    # scale for each element
    element_scale::Array{mk_float,1}
    # shift for each element
    element_shift::Array{mk_float,1}
    # list of element boundaries
    element_boundaries::Array{mk_float,1}
    # Lagrange Polynomial data for each element
    lpoly_data::Union{Array{lagrange_poly_data,1},Nothing}
    # Coordinate data for each element
    element_data::Union{Array{element_coordinates,1},Nothing}
    # data required to take a first derivative (only)
    derivative_data::Union{first_derivative_data,Nothing}
    """
    This internal constructor for `finite_element_coordinate`
    takes `scalar_input(ngrid,nelement,L)` to construct
    the `element_data` struct for the fundamental constructor.
    """
    function finite_element_coordinate(
        # name of coordinate
        name::String,
        # the inputs used to construct a FEM grid
        scalar_input::scalar_coordinate_inputs;
        # option for spacing between element boundaries
        element_spacing_option="uniform"::String,
        # which boundary condition to use
        bc=natural_boundary_condition::finite_element_boundary_condition_type)
        ngrid = scalar_input.ngrid
        nelement = scalar_input.nelement
        Ldomain = scalar_input.Ldomain
        # initialise the data used to construct the grid
        # boundaries for each element
        element_boundaries = set_element_boundaries(nelement,
                                                    Ldomain,
                                                    element_spacing_option,
                                                    name)
        # shift and scale factors for each local element
        element_scale, element_shift =
            set_element_scale_and_shift(element_boundaries)
        if ngrid > 1
            # get the nodes on [-1,1] for each element
            reference_nodes = reference_grids(ngrid,nelement,name)
            element_data = Array{element_coordinates,1}(undef,nelement)
            for ielement in 1:nelement
                # get the reference nodes defined on [-1,1] (or (-1,1] on radau elements))
                scale = element_scale[ielement]
                shift = element_shift[ielement]
                @views x_nodes = reference_nodes[:,ielement]
                element_data[ielement] = element_coordinates(x_nodes,
                                                        scale,
                                                        shift)
            end
        else
            element_data = nothing
        end
        return finite_element_coordinate(name, element_data, bc=bc)
    end
    """
    This is the fundamental internal constructor
    for `finite_element_coordinate`, which takes
    `element_data::Union{Array{element_coordinates,1},Nothing}`
    as an argument to define the grid.

    The option to pass a value with type `Nothing` is
    required to permit a trivial coordinate of 1 point
    to be constructed with this function.
    """
    function finite_element_coordinate(
        # name of coordinate
        name::String,
        # array containing data defining element grids
        # from which the coordinate struct can be created
        element_data::Union{Array{element_coordinates,1},Nothing};
        # which boundary condition to use
        bc=natural_boundary_condition::finite_element_boundary_condition_type)
        if typeof(element_data) == Nothing
            # this is a trivial coordinate of length 1
            nelement = 1
            ngrid = 1
            element_scale = ones(mk_float,1)
            element_shift = zeros(mk_float,1)
            element_boundaries = zeros(mk_float,2)
            element_boundaries[1] = -1.0
            element_boundaries[2] = 1.0
        else
            # number of elements
            nelement = length(element_data)
            # number of grid points per element
            ngrid = length(element_data[1].lpoly_data.x_nodes)
            # check ngrid the same for each element
            for j in 2:nelement
                if !(length(element_data[j].lpoly_data.x_nodes) == ngrid)
                    error("length(element_data[j].lpoly_data.x_nodes) /= ngrid \n Number of nodes in reference grid must be the same for each element")
                end
            end
            # extract shift, scale, and boundary values
            element_scale = allocate_float(nelement)
            element_shift = allocate_float(nelement)
            element_boundaries = allocate_float(nelement+1)
            for j in 1:nelement
                element_scale[j] = element_data[j].scale
                element_shift[j] = element_data[j].shift
                element_boundaries[j] = element_shift[j] - element_scale[j]
            end
            element_boundaries[nelement+1] = element_scale[nelement] + element_shift[nelement]
        end
        # total number of grid points is ngrid for the first element
        # plus ngrid-1 unique points for each additional element due
        # to the repetition of a point at the element boundary
        n_global = (ngrid-1)*nelement + 1
        # obtain index mapping from full (local) grid to the
        # grid within each element (igrid, ielement)
        igrid, ielement = full_to_elemental_grid_map(ngrid,
                            nelement, n_global)
        # obtain (local) index mapping from the grid within each element
        # to the full grid
        imin, imax, igrid_full = elemental_to_full_grid_map(ngrid,
                                                            nelement)
        # initialize the grid and the integration weights associated with the grid
        grid = allocate_float(n_global)
        wgts = zeros(mk_float,n_global)
        if n_global > 1
            nquad = 2*ngrid
            zz, wz = gausslegendre(nquad)
            k = 1
            for j in 1:nelement
                # extract reference nodes
                x_nodes = element_data[j].lpoly_data.x_nodes
                # calculate weights on the reference nodes
                wgts_nodes = allocate_float(ngrid)
                for i in 1:ngrid
                    ith_lpoly_data = element_data[j].lpoly_data.lpoly_data[i]
                    result = 0.0
                    for l in 1:nquad
                        result += wz[l]*lagrange_poly(ith_lpoly_data,zz[l])
                    end
                    wgts_nodes[i] = result
                end
                # put this data into the global arrays
                scale = element_scale[j]
                shift = element_shift[j]
                @. grid[igrid_full[k,j]:igrid_full[ngrid,j]] = x_nodes[k:ngrid]*scale + shift
                @. wgts[igrid_full[1,j]:igrid_full[ngrid,j]] += wgts_nodes[1:ngrid]*scale
                k = 2
            end
            # include Jacobian factor consistent with original grid
            if name == "vperp"
                @. wgts *= 2*pi*grid
            end
        else
            grid[1] = 0.0
            wgts[1] = 1.0
        end
        if ngrid > 1
            lpoly_data = Array{lagrange_poly_data,1}(undef,nelement)
            #element_data = Array{element_coordinates,1}(undef,nelement)
            #x_nodes = allocate_float(ngrid)
            for ielement in 1:nelement
                # get the local grid in global coord system
                grid_local = grid[igrid_full[1,ielement]:igrid_full[ngrid,ielement]]
                # get Lagrange Poly data for interpolating in global coordinates
                lpoly_data[ielement] = lagrange_poly_data(grid_local)
            end
        else
            lpoly_data = nothing
        end
        if typeof(element_data) == Nothing
            derivative_data = nothing
        else
            derivative_data = first_derivative_data(ngrid, nelement,
                                        n_global, igrid_full, element_data)
        end
        domainLength = element_boundaries[end] - element_boundaries[1]
        return new(name, n_global, ngrid,
            nelement, domainLength, grid, igrid, ielement, imin, imax,
            igrid_full, bc, wgts,
            element_scale, element_shift, element_boundaries,
            lpoly_data, element_data, derivative_data)
    end
end

function set_element_boundaries(nelement_global::mk_int, L::mk_float,
        element_spacing_option::String, coord_name::String)
    # set global element boundaries between [-L/2,L/2]
    element_boundaries = allocate_float(nelement_global+1)
    if element_spacing_option == "uniform" # uniform spacing
        for j in 1:nelement_global+1
            element_boundaries[j] = L*((j-1)/(nelement_global) - 0.5)
        end
    else 
        println("ERROR: element_spacing_option: ",element_spacing_option, " not supported")
    end
    if coord_name == "vperp"
        #shift so that the range of element boundaries is [0,L]
        for j in 1:nelement_global+1
            element_boundaries[j] += L/2.0
        end
    end
    return element_boundaries
end

function set_element_scale_and_shift(element_boundaries::Array{mk_float,1})
    nelement = length(element_boundaries) - 1
    element_scale = allocate_float(nelement)
    element_shift = allocate_float(nelement)
    for j in 1:nelement
        upper_boundary = element_boundaries[j+1]
        lower_boundary = element_boundaries[j]
        element_scale[j] = 0.5*(upper_boundary-lower_boundary)
        element_shift[j] = 0.5*(upper_boundary+lower_boundary)
    end
    return element_scale, element_shift
end

"""
Makes an array `reference_nodes` of shape `(ngrid,nelement)`
with `reference_nodes[:,j]` the nodes of the jth element
defined on the grid that goes from -1,1,
"""
function reference_grids(ngrid::mk_int,nelement::mk_int,name::String)
    reference_nodes = allocate_float(ngrid,nelement)
    # get Gauss-Legendre-Lobatto points and weights on [-1,1]
    x_lob, w_lob = gausslobatto(ngrid)
    # get Gauss-Legendre-Radau points and weights on [-1,1)
    x_rad, w_rad = gaussradau(ngrid)
    # transform to a Gauss-Legendre-Radau grid on (-1,1]
    x_rad, w_rad = -reverse(x_rad), reverse(w_rad)
    if name == "vperp"
        reference_nodes[:,1] .= x_rad
    else
        reference_nodes[:,1] .= x_lob
    end
    for j in 2:nelement
        reference_nodes[:,j] .= x_lob
    end
    return reference_nodes
end

"""
setup arrays containing a map from the unpacked grid point indices
to the element index and the grid point index within each element
"""
function full_to_elemental_grid_map(ngrid::mk_int,
                        nelement::mk_int, n::mk_int)
    igrid = allocate_int(n)
    ielement = allocate_int(n)
    k = 1
    for i ∈ 1:ngrid
        ielement[k] = 1
        igrid[k] = i
        k += 1
    end
    if nelement > 1
        for j ∈ 2:nelement
            # avoid double-counting overlapping point
            # at boundary between elements
            for i ∈ 2:ngrid
                ielement[k] = j
                igrid[k] = i
                k += 1
            end
        end
    end
    return igrid, ielement
end

"""
returns imin and imax, which contain the minimum and maximum
indices on the full grid for each element
"""
function elemental_to_full_grid_map(ngrid::mk_int, nelement::mk_int)
    imin = allocate_int(nelement)
    imax = allocate_int(nelement)
    igrid_full = allocate_int(ngrid, nelement)
    @inbounds begin
        # the first element contains ngrid entries
        imin[1] = 1
        imax[1] = ngrid
        # each additional element contributes ngrid-1 unique entries
        # due to repetition of one grid point at the boundary
        if nelement > 1
            for i ∈ 2:nelement
                imin[i] = imax[i-1] + 1
                imax[i] = imin[i] + ngrid - 2
            end
        end
        
        for j in 1:nelement
            for i in 1:ngrid
                igrid_full[i,j] = i + (j - 1)*(ngrid - 1)
            end
        end
    end
    return imin, imax, igrid_full
end

function first_derivative!(df::AbstractArray{mk_float,1},
                        f::AbstractArray{mk_float,1},
                        coord::finite_element_coordinate)
    derivative_data = coord.derivative_data
    dummy_rhs = derivative_data.dummy_rhs
    dummy_df = derivative_data.dummy_df
    PP1D = derivative_data.PP1D
    lu_MM1D = derivative_data.lu_MM1D
    mul!(dummy_rhs,PP1D,f)
    ldiv!(dummy_df,lu_MM1D,dummy_rhs)
    @. df = dummy_df
    return nothing
end

end
