module MyUtils

using DataFrames
import Plots

function Plots.plot(df::DataFrame; kwargs...)
    if ncol(df) != 2 && ncol(df) != 1
        error("number of DataFrame columns for plotting should be 1 or 2")
    end
    ncol(df) == 2 && return Plots.plot(df[:, 1], df[:, 2]; kwargs...)
    ncol(df) == 1 && return Plots.plot(df[:, 1]; kwargs...)
end

function Plots.plot!(df::DataFrame; kwargs...)
    if ncol(df) != 2 && ncol(df) != 1
        error("number of DataFrame columns for plotting should be 1 or 2")
    end
    ncol(df) == 2 && return Plots.plot!(df[:, 1], df[:, 2]; kwargs...)
    ncol(df) == 1 && return Plots.plot!(df[:, 1]; kwargs...)
end

using Dates
import Serialization
export safe_serialize, deserialize

function safe_serialize(fn::String, dat)
    fls = readdir()
    if findfirst(x->x==fn, fls) == nothing
        Serialization.serialize(fn, dat)
    else
        prefix = Dates.format(now(), "yyyy-mm-dd--HH-MM-SS-")
        println(stderr, "file \"$fn\" already exists. Saving as \"$(prefix * fn)\".")
        Serialization.serialize(prefix * fn, dat)
    end
end

function deserialize(fn::AbstractString)
    Serialization.deserialize(fn)
end

end