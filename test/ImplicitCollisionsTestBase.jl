using FokkerPlanck.array_allocation: allocate_float
using FokkerPlanck.type_definitions: mk_float, mk_int
using FokkerPlanck: calculate_entropy_production
using FokkerPlanck.coordinates: scalar_coordinate_inputs,
                            set_element_boundaries,
                            set_element_scale_and_shift,
                            finite_element_coordinate,
                            finite_element_boundary_condition_type,
                            zero_boundary_condition, natural_boundary_condition
using FokkerPlanck.fokker_planck_test: F_Maxwellian, F_Beam, print_test_data
using FokkerPlanck.velocity_moments: get_density, get_upar, get_pressure, get_ppar, get_qpar, get_rmom
using FokkerPlanck.fokker_planck_calculus: species_info
using FiniteElementMatrices: element_coordinates
using Printf

struct moments_struct
    density::Vector{mk_float}
    upar::Vector{mk_float}
    vth::Vector{mk_float}
    pressure::Vector{mk_float}
    ppar::Vector{mk_float}
    qpar::Vector{mk_float}
    rmom::Vector{mk_float}
    function moments_struct(nspecies::mk_int)
        density = allocate_float(nspecies)
        upar = allocate_float(nspecies)
        vth = allocate_float(nspecies)
        pressure = allocate_float(nspecies)
        ppar = allocate_float(nspecies)
        qpar = allocate_float(nspecies)
        rmom = allocate_float(nspecies)
        return new(density, upar, vth, pressure, ppar, qpar, rmom)
    end
end

function calculate_total_parallel_momentum(moments::moments_struct,species::species_info)
    parallel_momentum = 0.0
    for is in 1:species.n
        parallel_momentum += species.mass[is]*moments.density[is]*moments.upar[is]
    end
    return parallel_momentum
end

function calculate_total_energy(moments::moments_struct,species::species_info)
    total_energy = 0.0
    for is in 1:species.n
        total_energy += (0.5*species.mass[is]*moments.density[is]*(moments.upar[is]^2)
                           + 1.5*moments.pressure[is])
    end
    return total_energy
end

function get_moments(pdf::AbstractArray{mk_float,2},fkpl_arrays,mass::mk_float)
    # extract coordinates
    vpa = fkpl_arrays.vpa
    vperp = fkpl_arrays.vperp
    dens = get_density(pdf,vpa,vperp)
    upar = get_upar(pdf, vpa, vperp, dens)
    pressure = get_pressure(pdf, vpa, vperp, upar)
    vth = sqrt(2.0*pressure/(dens*mass))
    ppar = get_ppar(pdf, vpa, vperp, upar)
    qpar = get_qpar(pdf, vpa, vperp, upar)
    rmom = get_rmom(pdf, vpa, vperp, upar)
    return dens, upar, vth, pressure, ppar, qpar, rmom
end

function diagnose_F_Maxwellian(pdf::AbstractArray{mk_float,2},
                    pdf_exact::AbstractArray{mk_float,2},
                    pdf_dummy_1::AbstractArray{mk_float,2},
                    pdf_dummy_2::AbstractArray{mk_float,2},
                    fkpl_arrays::fokkerplanck_weakform_arrays_struct,
                    time::mk_float,
                    mass::mk_float,
                    it::mk_int)
    # extract coordinates
    vpa = fkpl_arrays.vpa
    vperp = fkpl_arrays.vperp
    dens, upar, vth, pressure, ppar, qpar, rmom = get_moments(pdf,fkpl_arrays,mass)
    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                pdf_exact[ivpa,ivperp] = F_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
            end
        end
    end
    println("it = ", it, " time: ", time)
    print_test_data(pdf_exact,pdf,pdf_dummy_1,"F",vpa,vperp,pdf_dummy_2;print_to_screen=true)
    println("dens: ", dens)
    println("upar: ", upar)
    println("vth: ", vth)
    println("ppar: ", ppar)
    println("qpar: ", qpar)
    println("rmom: ", rmom)
    dSdt = calculate_entropy_production(pdf,fkpl_arrays)
    println("dSdt: ", dSdt)
    if vpa.bc == zero_boundary_condition
        println("test vpa bc: F[1, :]", pdf[1, :])
        println("test vpa bc: F[end, :]", pdf[end, :])
    end
    if vperp.bc == zero_boundary_condition
        println("test vperp bc: F[:, end]", pdf[:, end])
    end
end
function diagnose_F_Maxwellian(pdf::AbstractArray{mk_float,3},
                    pdf_exact::AbstractArray{mk_float,3},
                    pdf_dummy_1::AbstractArray{mk_float,2},
                    pdf_dummy_2::AbstractArray{mk_float,2},
                    fkpl_arrays::fokkerplanck_weakform_arrays_struct,
                    moments::moments_struct,
                    time::mk_float,
                    it::mk_int)
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
            moments.ppar[is],
            moments.qpar[is],
            moments.rmom[is] = @views get_moments(pdf[:,:,is],fkpl_arrays,species.mass[is])
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    pdf_exact[ivpa,ivperp,is] = F_Maxwellian(moments.density[is],
                                                            moments.upar[is],
                                                            moments.vth[is],
                                                            vpa,vperp,ivpa,ivperp)
                end
            end
        end
    end
    println("it = ", it, " time: ", time)
    for is in 1:species.n
        @views print_test_data(pdf_exact[:,:,is],pdf[:,:,is],pdf_dummy_1,"F[$is]",vpa,vperp,pdf_dummy_2;print_to_screen=true)
    end
    # println("upar: ", moments.upar)
    # println("vth: ", moments.vth)
    # println("ppar: ", moments.ppar)
    # println("qpar: ", moments.qpar)
    # println("rmom: ", moments.rmom)
    total_parallel_momentum = calculate_total_parallel_momentum(moments,species)
    total_energy = calculate_total_energy(moments,species)
    dSdt = calculate_entropy_production(pdf,fkpl_arrays)
    println("dens: ", moments.density)
    println("parallel momentum: ", total_parallel_momentum)
    println("total energy: ", total_energy)
    println("dSdt: ", dSdt)
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

function chebyshevpoints(n::mk_int;radau=false)
    grid = allocate_float(n)
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
                    input::scalar_coordinate_inputs)
    ngrid = input.ngrid
    nelement = input.nelement
    Ldomain = input.Ldomain
    # set vpa domain to be [-Ldomain/2,Ldomain/2], or set vperp domain to be [0, Ldomain]
    element_boundaries = set_element_boundaries(nelement, Ldomain, "uniform", name)
    # extract transformation factors such that v = scale * x + shift
    # with x the local reference grid value in [-1,1] (or (-1,1] for Radau elements)).
    element_scale, element_shift = set_element_scale_and_shift(element_boundaries)
    # reference Chebyshev grind points in (-1,1], [-1,1].
    grid_radau = chebyshevpoints(ngrid,radau=true)
    grid_lobatto = chebyshevpoints(ngrid)
    # construct the struct that contains the information
    # needed for FokkerPlanck to construct the internal vpa vperp grids.
    element_data = Array{element_coordinates,1}(undef,nelement)
    if name == "vperp"
        grid_low = grid_radau
    else
        grid_low = grid_lobatto
    end
    element_data[1] = element_coordinates(grid_low, element_scale[1], element_shift[1])
    for j in 2:nelement
        element_data[j] = element_coordinates(grid_lobatto,element_scale[j],element_shift[j])
    end
    return element_data
end

function set_initial_pdf!(Fold::AbstractArray{mk_float,2},
            vpa::finite_element_coordinate,
            vperp::finite_element_coordinate,
            vpa0::mk_float,
            vperp0::mk_float,
            vth0::mk_float,
            zbeam::mk_float)
    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                Fold[ivpa,ivperp] = F_Beam(vpa0,vperp0,vth0,vpa,vperp,ivpa,ivperp) +
                                    + zbeam * F_Beam(0.0,vperp0,vth0,vpa,vperp,ivpa,ivperp)
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

function print_pdf(pdf::AbstractArray{mk_float,3})
    println("# Expected Fout")
    print("[")
    nvpa, nvperp, ntind = size(pdf)
    for k in 1:ntind
        for i in 1:nvpa-1
            for j in 1:nvperp-1
                @printf("%.15f ", pdf[i,j,k])
            end
            @printf("%.15f ", pdf[i,nvperp,k])
            print(";\n")
        end
        for j in 1:nvperp-1
            @printf("%.15f ", pdf[nvpa,j,k])
        end
        @printf("%.15f ", pdf[nvpa,nvperp,k])
        if k < ntind
            print(";;;\n")
        end
    end
    print("]\n")
    return nothing
end

struct pdf_and_grid
    vpa_grid::Vector{mk_float}
    vperp_grid::Vector{mk_float}
    pdf::AbstractArray{mk_float,3}
end