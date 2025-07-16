"""
Module to provide methods for functions JacobianFreeNewtonKrylov,
and to import functions from JacobianFreeNewtonKrylov.
"""
module fokker_planck_nonlinear_solvers

export setup_fp_nl_solve

import JacobianFreeNewtonKrylov: distributed_norm,
                                 distributed_dot,
                                 parallel_map,
                                 parallel_delta_x_calc
using JacobianFreeNewtonKrylov: nl_solver_info
using ..coordinates: finite_element_coordinate
using ..type_definitions: mk_float, mk_int

function distributed_norm(
                               ::Val{:vperpvpa},
                               residual::AbstractArray{mk_float, 2},
                               coords, rtol, atol, x::AbstractArray{mk_float, 2})
    pdf_residual = residual
    x_pdf = x
    vperp = coords.vperp
    vpa = coords.vpa

    pdf_norm_square = 0.0
    @inbounds begin
        for ivperp in 1:vperp.n
             for ivpa in 1:vpa.n
                pdf_norm_square += (pdf_residual[ivpa,ivperp] / (rtol * abs(x_pdf[ivpa,ivperp]) + atol))^2
             end
        end
    end
    global_norm = Ref(pdf_norm_square)
    
    global_norm[] = sqrt(global_norm[] / (vperp.n * vpa.n))
    
    return global_norm[]
end

function distributed_dot(
                  ::Val{:vperpvpa}, v::AbstractArray{mk_float, 2},
                  w::AbstractArray{mk_float, 2}, coords,
                  rtol, atol, x::AbstractArray{mk_float, 2})
    v_pdf = v
    w_pdf = w
    x_pdf = x

    vperp = coords.vperp
    vpa = coords.vpa

    pdf_dot = 0.0
    @inbounds begin
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                pdf_dot += v_pdf[ivpa,ivperp] * w_pdf[ivpa,ivperp] / (rtol * abs(x_pdf[ivpa,ivperp]) + atol)^2
            end
        end
    end
    global_dot = Ref(pdf_dot)
    global_dot[] = global_dot[] / (vperp.n * vpa.n)
    return global_dot[]
end

function parallel_map(
                  ::Val{:vperpvpa}, func, result::AbstractArray{mk_float, 2})

    result_pdf = result
    nvpa, nvperp = size(result)
    @inbounds begin
        for ivperp in 1:nvperp
            for ivpa in 1:nvpa
                result_pdf[ivpa,ivperp] = func()
            end
        end
    end
    return nothing
end
function parallel_map(
                  ::Val{:vperpvpa}, func, result::AbstractArray{mk_float, 2},
                  x1)

    result_pdf = result
    x1_pdf = x1
    nvpa, nvperp = size(result)
    @inbounds begin
        for ivperp in 1:nvperp
            for ivpa in 1:nvpa
                result_pdf[ivpa,ivperp] = func(x1_pdf[ivpa,ivperp])
            end
        end
    end
    return nothing
end
function parallel_map(
                  ::Val{:vperpvpa}, func, result::AbstractArray{mk_float, 2},
                  x1, x2)

    result_pdf = result
    x1_pdf = x1
    nvpa, nvperp = size(result)
    if isa(x2, AbstractArray)
        x2_pdf = x2
    
        @inbounds begin
            for ivperp in 1:nvperp
                for ivpa in 1:nvpa
                    result_pdf[ivpa,ivperp] = func(x1_pdf[ivpa,ivperp], x2_pdf[ivpa,ivperp])
                end
            end
        end
    else
        @inbounds begin
            for ivperp in 1:nvperp
                for ivpa in 1:nvpa
                    result_pdf[ivpa,ivperp] = func(x1_pdf[ivpa,ivperp], x2)
                end
            end
        end
    end
    return nothing
end
function parallel_map(
                  ::Val{:vperpvpa}, func, result::AbstractArray{mk_float, 2},
                  x1, x2, x3)

    result_pdf = result
    x1_pdf = x1
    x2_pdf = x2
    nvpa, nvperp = size(result)
    if isa(x3, AbstractArray)
        x3_pdf = x3
    
        @inbounds begin
            for ivperp in 1:nvperp
                for ivpa in 1:nvpa
                    result_pdf[ivpa,ivperp] = func(x1_pdf[ivpa,ivperp], x2_pdf[ivpa,ivperp], x3_pdf[ivpa,ivperp])
                end
            end
        end
    else
        @inbounds begin
            for ivperp in 1:nvperp
                for ivpa in 1:nvpa
                    result_pdf[ivpa,ivperp] = func(x1_pdf[ivpa,ivperp], x2_pdf[ivpa,ivperp], x3)
                end
            end
        end
    end
    return nothing
end

function parallel_delta_x_calc(
                  ::Val{:vperpvpa}, delta_x::AbstractArray{mk_float, 2}, V,
                  y)

    delta_x_pdf = delta_x
    V_pdf = V

    ny = length(y)
    nvpa, nvperp = size(delta_x)
    @inbounds begin
        for iy ∈ 1:ny
            for ivperp in 1:nvperp
                for ivpa in 1:nvpa
                    delta_x_pdf[ivpa,ivperp] += y[iy] * V_pdf[ivpa,ivperp,iy]
                end
            end
        end
    end

    return nothing
end

"""
Function to setup nonlinear_solver struct for implicit
Fokker-Planck collisions. Wrapper to avoid extra imports
of nonlinear_solvers.jl.
"""
function setup_fp_nl_solve(vpa::finite_element_coordinate,
                           vperp::finite_element_coordinate;
                           kwargs...)
    coords = (vperp=vperp,vpa=vpa)
    return nl_solver_info(coords; kwargs...)
end


end