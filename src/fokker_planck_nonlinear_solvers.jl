"""
Module to provide methods for functions JacobianFreeNewtonKrylov,
and to import functions from JacobianFreeNewtonKrylov.
"""
module fokker_planck_nonlinear_solvers

import JacobianFreeNewtonKrylov: distributed_norm,
                                 distributed_dot,
                                 parallel_map,
                                 parallel_delta_x_calc
using JacobianFreeNewtonKrylov: nl_solver_info
using FiniteElementAssembly: FiniteElementCoordinate

function distributed_norm(
                               ::Val{:speciesvperpvpa},
                               residual::AbstractArray{Float64, 3},
                               coords, rtol, atol, x::AbstractArray{Float64, 3})
    pdf_residual = residual
    x_pdf = x
    species = coords.species
    vperp = coords.vperp
    vpa = coords.vpa

    pdf_norm_square = 0.0
    @inbounds begin
        for is in 1:species.n
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    pdf_norm_square += (pdf_residual[ivpa,ivperp,is] / (rtol * abs(x_pdf[ivpa,ivperp,is]) + atol))^2
                end
            end
        end
    end
    global_norm = Ref(pdf_norm_square)
    global_norm[] = sqrt(global_norm[] / (species.n * vperp.n * vpa.n))
    return global_norm[]
end

function distributed_dot(
                  ::Val{:speciesvperpvpa}, v::AbstractArray{Float64, 3},
                  w::AbstractArray{Float64, 3}, coords,
                  rtol, atol, x::AbstractArray{Float64, 3})
    v_pdf = v
    w_pdf = w
    x_pdf = x
    species = coords.species
    vperp = coords.vperp
    vpa = coords.vpa

    pdf_dot = 0.0
    @inbounds begin
        for is in 1:species.n
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    pdf_dot += v_pdf[ivpa,ivperp,is] * w_pdf[ivpa,ivperp,is] / (rtol * abs(x_pdf[ivpa,ivperp,is]) + atol)^2
                end
            end
        end
    end
    global_dot = Ref(pdf_dot)
    global_dot[] = global_dot[] / (species.n * vperp.n * vpa.n)
    return global_dot[]
end

function parallel_map(
                  ::Val{:speciesvperpvpa}, func, result::AbstractArray{Float64, 3})

    result_pdf = result
    nvpa, nvperp, nspecies = size(result)
    @inbounds begin
        for is in 1:nspecies
            for ivperp in 1:nvperp
                for ivpa in 1:nvpa
                    result_pdf[ivpa,ivperp,is] = func()
                end
            end
        end
    end
    return nothing
end
function parallel_map(
                  ::Val{:speciesvperpvpa}, func, result::AbstractArray{Float64, 3},
                  x1)

    result_pdf = result
    x1_pdf = x1
    nvpa, nvperp, nspecies = size(result)
    @inbounds begin
        for is in 1:nspecies
            for ivperp in 1:nvperp
                for ivpa in 1:nvpa
                    result_pdf[ivpa,ivperp,is] = func(x1_pdf[ivpa,ivperp,is])
                end
            end
        end
    end
    return nothing
end
function parallel_map(
                  ::Val{:speciesvperpvpa}, func, result::AbstractArray{Float64, 3},
                  x1, x2)

    result_pdf = result
    x1_pdf = x1
    nvpa, nvperp, nspecies = size(result)
    if isa(x2, AbstractArray)
        x2_pdf = x2
        @inbounds begin
            for is in 1:nspecies
                for ivperp in 1:nvperp
                    for ivpa in 1:nvpa
                        result_pdf[ivpa,ivperp,is] = func(x1_pdf[ivpa,ivperp,is], x2_pdf[ivpa,ivperp,is])
                    end
                end
            end
        end
    else
        @inbounds begin
            for is in 1:nspecies
                for ivperp in 1:nvperp
                    for ivpa in 1:nvpa
                        result_pdf[ivpa,ivperp,is] = func(x1_pdf[ivpa,ivperp,is], x2)
                    end
                end
            end
        end
    end
    return nothing
end
function parallel_map(
                  ::Val{:speciesvperpvpa}, func, result::AbstractArray{Float64, 3},
                  x1, x2, x3)

    result_pdf = result
    x1_pdf = x1
    x2_pdf = x2
    nvpa, nvperp, nspecies = size(result)
    if isa(x3, AbstractArray)
        x3_pdf = x3
        @inbounds begin
            for is in 1:nspecies
                for ivperp in 1:nvperp
                    for ivpa in 1:nvpa
                        result_pdf[ivpa,ivperp,is] = func(x1_pdf[ivpa,ivperp,is], x2_pdf[ivpa,ivperp,is], x3_pdf[ivpa,ivperp,is])
                    end
                end
            end
        end
    else
        @inbounds begin
            for is in 1:nspecies
                for ivperp in 1:nvperp
                    for ivpa in 1:nvpa
                        result_pdf[ivpa,ivperp,is] = func(x1_pdf[ivpa,ivperp,is], x2_pdf[ivpa,ivperp,is], x3)
                    end
                end
            end
        end
    end
    return nothing
end

function parallel_delta_x_calc(
                  ::Val{:speciesvperpvpa}, delta_x::AbstractArray{Float64, 3}, V,
                  y)

    delta_x_pdf = delta_x
    V_pdf = V

    ny = length(y)
    nvpa, nvperp, nspecies = size(delta_x)
    @inbounds begin
        for iy ∈ 1:ny
            for is in 1:nspecies
                for ivperp in 1:nvperp
                    for ivpa in 1:nvpa
                        delta_x_pdf[ivpa,ivperp,is] += y[iy] * V_pdf[ivpa,ivperp,is,iy]
                    end
                end
            end
        end
    end

    return nothing
end


end