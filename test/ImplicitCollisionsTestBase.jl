using FokkerPlanck: calculate_entropy_production
using FiniteElementAssembly: ScalarCoordinateInputs,
                            set_element_boundaries,
                            set_element_scale_and_shift,
                            FiniteElementCoordinate, AbstractBoundaryCondition,
                            include_boundary_points, exclude_lower_boundary_point
using FokkerPlanck.fokker_planck_calculus: zero_boundary_condition, natural_boundary_condition
using FokkerPlanck.fokker_planck_test: F_Maxwellian, F_Beam, print_test_data
using FokkerPlanck.velocity_moments: get_density, get_upar, get_pressure, get_ppar, get_qpar, get_rmom
using FokkerPlanck.fokker_planck_calculus: SpeciesData
using FokkerPlanck: ElementCoordinates # from FiniteElementMatrices, re-exported via FokkerPlanck
using Printf

struct moments_struct
    density::Vector{Float64}
    upar::Vector{Float64}
    vth::Vector{Float64}
    pressure::Vector{Float64}
    temperature::Vector{Float64}
    ppar::Vector{Float64}
    qpar::Vector{Float64}
    rmom::Vector{Float64}
    conserved::Vector{Float64}
    function moments_struct(nspecies::Int64)
        density = Array{Float64}(undef,nspecies)
        upar = Array{Float64}(undef,nspecies)
        vth = Array{Float64}(undef,nspecies)
        pressure = Array{Float64}(undef,nspecies)
        temperature = Array{Float64}(undef,nspecies)
        ppar = Array{Float64}(undef,nspecies)
        qpar = Array{Float64}(undef,nspecies)
        rmom = Array{Float64}(undef,nspecies)
        conserved = Array{Float64}(undef,nspecies+2)
        return new(density, upar, vth, pressure,
            temperature, ppar, qpar, rmom, conserved)
    end
end

function calculate_total_parallel_momentum(moments::moments_struct,species::SpeciesData)
    parallel_momentum = 0.0
    for is in 1:species.n
        parallel_momentum += species.mass[is]*moments.density[is]*moments.upar[is]
    end
    return parallel_momentum
end

function calculate_total_energy(moments::moments_struct,species::SpeciesData)
    total_energy = 0.0
    for is in 1:species.n
        total_energy += (0.5*species.mass[is]*moments.density[is]*(moments.upar[is]^2)
                           + 1.5*moments.pressure[is])
    end
    return total_energy
end
function calculate_total_change(CCs,fkpl_arrays)
    vpa = fkpl_arrays.vpa
    vperp = fkpl_arrays.vperp
    species = fkpl_arrays.species
    mass = species.mass
    c0ref = species.c0ref
    u0ref = species.u0ref
    n0ref = species.n0ref
    total_momentum_change = 0.0
    total_energy_change = 0.0
    for is in 1:species.n
        @views total_momentum_change += mass[is]*n0ref[is]*(c0ref[is]*get_upar(CCs[:,:,is],vpa,vperp,1.0)
                                            + u0ref[is]*get_density(CCs[:,:,is],vpa,vperp))
        up0 = (0.0 - u0ref[is])/c0ref[is] # velocity in reference frame of species s corresponding to lab frame u=0
        @views total_energy_change += 3.0*n0ref[is]*(c0ref[is]^2)*get_pressure(CCs[:,:,is],vpa,vperp,up0,mass[is])
    end
    return total_momentum_change, total_energy_change
end

function get_moments(pdf::Tpdf,
    vpa::FiniteElementCoordinate,vperp::FiniteElementCoordinate,
    mass::Float64,c0ref::Float64,u0ref::Float64,n0ref::Float64
    ) where Tpdf <: AbstractArray{Float64,2}
    dens = get_density(pdf,vpa,vperp)
    if abs(dens) < 1.0e-14
        density, upar, pressure, temperature, vth, ppar, qpar, rmom = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    else
        density = n0ref*dens
        up0 = get_upar(pdf, vpa, vperp, dens)
        upar = c0ref*up0 + u0ref
        pressure = n0ref*(c0ref^2)*get_pressure(pdf, vpa, vperp, up0, mass)
        temperature = pressure/density
        vth = sqrt(2.0*temperature/mass)
        ppar = n0ref*(c0ref^2)*get_ppar(pdf, vpa, vperp, up0, mass)
        qpar = n0ref*(c0ref^3)*get_qpar(pdf, vpa, vperp, up0, mass)
        rmom = n0ref*(c0ref^4)*get_rmom(pdf, vpa, vperp, up0, mass)
    end
    return density, upar, vth, pressure, temperature, ppar, qpar, rmom
end

function diagnose_F_Maxwellian(CC::Tpdf1, pdf::Tpdf2, pdf_exact::Tpdf3,
                    pdf_dummy_1::Tpdf4, pdf_dummy_2::Tpdf5,
                    fkpl_arrays::FokkerPlanckWeakformArrays,
                    moments::moments_struct,
                    time::Float64, it::Int64; updated_CC=true
                    ) where {Tpdf1 <: AbstractArray{Float64,3},Tpdf2 <: AbstractArray{Float64,3},
                    Tpdf3 <: AbstractArray{Float64,3}, Tpdf4 <: AbstractArray{Float64,2},
                    Tpdf5 <: AbstractArray{Float64,2}}
    # extract coordinates
    vpa = fkpl_arrays.vpa
    vperp = fkpl_arrays.vperp
    species = fkpl_arrays.species
    @inbounds begin
        for is in 1:species.n
            moments.density[is],
            moments.upar[is],
            moments.vth[is],
            moments.pressure[is],
            moments.temperature[is],
            moments.ppar[is],
            moments.qpar[is],
            moments.rmom[is] = @views get_moments(pdf[:,:,is],vpa,vperp,species.mass[is],
                                        species.c0ref[is],species.u0ref[is],species.n0ref[is])
            prefactor = (species.c0ref[is]^3)/species.n0ref[is]
            for ivperp in 1:vperp.n
                vperp_val = species.c0ref[is]*vperp.grid[ivperp]
                for ivpa in 1:vpa.n
                    vpa_val = species.c0ref[is]*vpa.grid[ivpa] + species.u0ref[is]
                    pdf_exact[ivpa,ivperp,is] = prefactor*F_Maxwellian(moments.density[is],
                                                            moments.upar[is],
                                                            moments.vth[is],
                                                            vpa_val,vperp_val)
                end
            end
        end
    end
    println("it = ", it, " time: ", time)
    for is in 1:species.n
        @views print_test_data(pdf_exact[:,:,is],pdf[:,:,is],pdf_dummy_1,"F[$is]_Maxwellian",vpa,vperp,pdf_dummy_2;print_to_screen=true)
    end
    # println("upar: ", moments.upar)
    # println("vth: ", moments.vth)
    # println("ppar: ", moments.ppar)
    # println("qpar: ", moments.qpar)
    # println("rmom: ", moments.rmom)
    total_parallel_momentum = calculate_total_parallel_momentum(moments,species)
    total_energy = calculate_total_energy(moments,species)
    dSdt = calculate_entropy_production(CC,pdf,fkpl_arrays)
    delta_momentum_C, delta_energy_C = calculate_total_change(CC,fkpl_arrays)
    if it == 0
        # store conserved quantities
        moments.conserved[1:species.n] .= moments.density
        moments.conserved[species.n+1] = total_parallel_momentum
        moments.conserved[species.n+2] = total_energy
    end
    println("dens: ", moments.density)
    println("upar: ", moments.upar)
    println("temp: ", moments.temperature)
    println("parallel momentum: ", total_parallel_momentum)
    println("total energy: ", total_energy)
    println("delta density: ", moments.density .- moments.conserved[1:species.n])
    println("delta momentum: ", total_parallel_momentum - moments.conserved[species.n+1])
    println("delta energy: ", total_energy - moments.conserved[species.n+2])
    if updated_CC
        println("dSdt: ", dSdt)
        println("delta_momentum_C: ",delta_momentum_C)
        println("delta_energy_C: ", delta_energy_C)
    end
    if vpa.bc == zero_boundary_condition
        for is in 1:species.n
            println("test vpa bc: F[1, :, $is]", pdf[1, :, is])
            println("test vpa bc: F[end, :, $is]", pdf[end, :, is])
        end
    end
    if vperp.bc == zero_boundary_condition
        for is in 1:species.n
            println("test vperp bc: F[:, end, $is]", pdf[:, end, is])
        end
    end
end

function chebyshevpoints(n::Int64;radau=false)
    grid = Array{Float64}(undef,n)
    if radau # exclude lower endpoint, grid ∈ (-1,1]
        nfac = 1.0/(n-0.5)
    else # include endpoints, grid ∈ [-1,1]
        nfac = 1.0/(n-1.0)
    end
    @inbounds begin
        for j ∈ 1:n
            grid[j] = cospi((n-j)*nfac)
        end
    end
    return grid
end

function chebyshev_grid(name::String,
                    input::ScalarCoordinateInputs)
    ngrid = input.ngrid
    nelement = input.nelement
    # set vpa domain to be [-Ldomain/2,Ldomain/2], or set vperp domain to be [0, Ldomain]
    element_boundaries = set_element_boundaries(input)
    # extract transformation factors such that v = scale * x + shift
    # with x the local reference grid value in [-1,1] (or (-1,1] for Radau elements)).
    element_scale, element_shift = set_element_scale_and_shift(element_boundaries)
    # reference Chebyshev grind points in (-1,1], [-1,1].
    grid_radau = chebyshevpoints(ngrid,radau=true)
    grid_lobatto = chebyshevpoints(ngrid)
    # construct the struct that contains the information
    # needed for FokkerPlanck to construct the internal vpa vperp grids.
    element_data = Array{ElementCoordinates,1}(undef,nelement)
    if name == "vperp"
        grid_low = grid_radau
    else
        grid_low = grid_lobatto
    end
    element_data[1] = ElementCoordinates(grid_low, element_scale[1], element_shift[1])
    for j in 2:nelement
        element_data[j] = ElementCoordinates(grid_lobatto,element_scale[j],element_shift[j])
    end
    return element_data
end

function set_initial_pdf!(Fold::Tpdf,
            vpa::FiniteElementCoordinate,
            vperp::FiniteElementCoordinate,
            vpa0::Float64,
            vperp0::Float64,
            vth0::Float64,
            zbeam::Float64) where Tpdf <: AbstractArray{Float64,2}
    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                Fold[ivpa,ivperp] = F_Beam(vpa0,vperp0,vth0,vpa.grid[ivpa],vperp.grid[ivperp]) +
                                    + zbeam * F_Beam(0.0,vperp0,vth0,vpa.grid[ivpa],vperp.grid[ivperp])
            end
        end
    end
    if vpa.bc == zero_boundary_condition
        @inbounds for ivperp in 1:vperp.n
            Fold[1,ivperp] = 0.0
            Fold[end,ivperp] = 0.0
        end
    end
    if vperp.bc == zero_boundary_condition
        @inbounds for ivpa in 1:vpa.n
            Fold[ivpa,end] = 0.0
        end
    end
    # normalise to unit density
    @views densfac = get_density(Fold,vpa,vperp)
    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                Fold[ivpa,ivperp] /= densfac
            end
        end
    end
    return nothing
end

function print_grid(coord)
    println("# Expected "*coord.name)
    print("[")
    for k in 1:coord.n
        @printf("%.15f", coord.grid[k])
        if k < coord.n
            print(", ")
        end
    end
    print("],\n")
    return nothing
end

function print_pdf(pdf::Tpdf) where Tpdf <: AbstractArray{Float64,4}
    println("# Expected Fout")
    print("[")
    nvpa, nvperp, nspecies, ntind = size(pdf)
    for l in 1:ntind
        for k in 1:nspecies
            for i in 1:nvpa-1
                for j in 1:nvperp-1
                    @printf("%.15f ", pdf[i,j,k,l])
                end
                @printf("%.15f ", pdf[i,nvperp,k,l])
                print(";\n")
            end
            for j in 1:nvperp-1
                @printf("%.15f ", pdf[nvpa,j,k,l])
            end
            @printf("%.15f ", pdf[nvpa,nvperp,k,l])
            if k < nspecies
                print(";;;\n")
            end
        end
        if l < ntind
            print(";;;;\n")
        end
    end
    print("]\n")
    return nothing
end

struct pdf_and_grid
    vpa_grid::Vector{Float64}
    vperp_grid::Vector{Float64}
    pdf::Array{Float64,4}
end