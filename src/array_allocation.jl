"""
"""
module array_allocation

export allocate_float, allocate_int

using ..type_definitions: mk_float, mk_int

"""
allocate 1d array with dimensions given by dims and entries of type mk_int
"""
function allocate_int(dims...)
    return array = Array{mk_int}(undef, dims...)
end

"""
allocate array with dimensions given by dims and entries of type mk_float
"""
function allocate_float(dims...)
    array = Array{mk_float}(undef, dims...)
    return array
end

end
