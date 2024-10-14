module YaoUtils

using Yao, DataFrames, LinearAlgebra, Memoization
export Rxx, Ryy, Rzz, repeated_rot, grad, grads, layered_repeated_rot, set_params
export set_apply, exact_st_evolution, bell_pairs, exact_op_evolution
export tr_mat_CJ, mat_CJ

Rxx(θ) = rot(kron(X,X), θ)
Ryy(θ) = rot(kron(Y,Y), θ)
Rzz(θ) = rot(kron(Z,Z), θ)

function Yao.print_block(io::IO, gt::RotationGate{2, T, KronBlock{2, 2, Tuple{P, P}}}) where {T<:Number, P<:Union{XGate, YGate, ZGate}}
    angle = if gt.theta isa Union{Float16, Float32, Float64}
        string(round(gt.theta, digits=2))
    else
        string(gt.theta)
    end
    gatename = if P<:XGate
        "Rxx"
    elseif P<:YGate
        "Ryy"
    elseif P<:ZGate
        "Rzz"
    else
        "N/A"
    end
    print(io, "$gatename($(angle))\n")
end

PauliType = Union{XGate, YGate, ZGate}

mutable struct RepeatedRotation{P <: PauliType, G <: Union{P, KronBlock{2, 2, Tuple{P, P}}}, R <: Real, N} <: CompositeBlock{2}
    n::Int64
    block::G
    locs::Vector{NTuple{N, Int64}}
    θ::R
end

mutable struct LayeredRepeatedRotation{R<:Real} <: CompositeBlock{2}
    n::Int64
    blocks::Vector
    θ::R
end

function repeated_rot(nq::Int64, block::G, locs::AbstractVector{Int64}, θ::Real) where {G <: PauliType}
    RepeatedRotation(nq, block, map(i->(i,), locs), θ)
end

function repeated_rot(nq::Int64, pair::Pair{V,G}, θ::Real) where {V<:AbstractVector{Int64}, G<:PauliType}
    block = pair.second
    locs = pair.first
    RepeatedRotation(nq, block, map(i->(i,), locs), θ)
end

function repeated_rot(nq::Int64, block::G, locs::AbstractVector{Tuple{Int64, Int64}}, θ::Real) where {
        P <: PauliType, G <: KronBlock{2, 2, Tuple{P, P}}}
    RepeatedRotation(nq, block, locs, θ)
end

function repeated_rot(nq::Int64, pair::Pair{V,G}, θ::Real) where {
        V<:AbstractVector{Tuple{Int64, Int64}}, P <: PauliType, G <: KronBlock{2, 2, Tuple{P, P}}}
    block = pair.second
    locs = pair.first
    RepeatedRotation(nq, block, locs, θ)
end

function layered_repeated_rot(rrot::Vector, θ::Real)
    if !prod(([r.n for r in rrot] .== rrot[1].n))
        error("")
    end
    LayeredRepeatedRotation(rrot[1].n, rrot, θ)
end

Yao.nqudits(g::RepeatedRotation) = g.n
Yao.nqudits(g::LayeredRepeatedRotation) = g.n

Yao.nparameters(::RepeatedRotation) = 1
Yao.nparameters(::LayeredRepeatedRotation) = 1

function Yao.occupied_locs(g::RepeatedRotation)
    s = Set{Int64}[]
    for i in g.locs
        push!(s, i...)
    end
    s |> collect |> sort! |> Tuple
end


function YaoBlocks.Optimise.to_basictypes(g::RepeatedRotation)
    r = rot(g.block, g.θ)
    chain(g.n, [put(i=>r) for i in g.locs])
end

function YaoBlocks.Optimise.to_basictypes(g::LayeredRepeatedRotation)
    for rrot in g.blocks
        setiparams!(rrot, g.θ)
    end
    chain(g.n, [g.blocks]...)
end

function Yao.setiparams!(g::RepeatedRotation, θ::Real)
    g.θ = θ
    g
end

function Yao.setiparams!(g::LayeredRepeatedRotation, θ::Real)
    g.θ = θ
    g
end

function Yao.unsafe_apply!(r::AbstractRegister, g::RepeatedRotation)
    cir = YaoBlocks.Optimise.to_basictypes(g)
    Yao.unsafe_apply!(r, cir)
end

function Yao.unsafe_apply!(r::AbstractRegister, g::LayeredRepeatedRotation)
    cir = YaoBlocks.Optimise.to_basictypes(g)
    Yao.unsafe_apply!(r, cir)
end

function grad(gt::RotationGate)
    gt * gt.block * (-im/2)
end

function grad(gt::RepeatedRotation)
    gt * +([chain(gt.n, put(i=>gt.block)) for i in gt.locs]...) * (-im/2)
end

function grad(gt::LayeredRepeatedRotation)
    cir = YaoBlocks.Optimise.to_basictypes(gt)
    ret = []
    for i in 1:length(gt.blocks)
        c = copy(cir)
        c.blocks[i] = grad(c.blocks[i])
        push!(ret, c)
    end
    sum(ret)
end

function YaoPlots.draw!(c::YaoPlots.CircuitGrid, p::Scale, address::Vector, controls::Vector)
    println("Scale[", p.alpha, "]")
    YaoPlots.draw!(c, p.content, address, controls)
end

function Yao.subblocks(g::RepeatedRotation)
    (rot(g.block, g.θ),)
end

function Yao.subblocks(gt::LayeredRepeatedRotation)
    [rot(g.block, gt.θ) for g in gt.blocks]
end

function grads(cir::ChainBlock)
    ret = ChainBlock{2}[]
    bls = cir.blocks
    for i in 1:length(bls)
        if bls[i] isa PutBlock
            if bls[i].content isa RotationGate
                ps = copy(cir)
                ps.blocks[i] = put(bls[i].n, bls[i].locs => grad(ps.blocks[i].content))
                push!(ret, ps)
            else
                error("")
            end
        elseif bls[i] isa Union{RepeatedRotation, LayeredRepeatedRotation}
            ps = copy(cir)
            ps.blocks[i] = grad(ps.blocks[i])
            push!(ret, ps)
        end
    end
    ret
end

function set_params(cir::ChainBlock, θ)
    if length(θ) != nparameters(cir)
        error("number of circuit parameters and length(θ) not match")
    end
    ret = deepcopy(cir)
    blks = ret.blocks
    for i in 1:length(blks)
        if !(blks[i] isa Union{RepeatedRotation, LayeredRepeatedRotation})
            error("blocks of ChainBlock must be RepeatedRotation or LayeredRepeatedRotation")
        end
         Yao.setiparams!(blks[i], θ[i])
    end
    ret
end

function set_apply(init::AbstractRegister, cir::ChainBlock, θ)
    cir2 = set_params(cir, θ)
    apply(init, cir2)
end

function exact_st_evolution(init::AbstractArrayReg, hamilt::AbstractBlock, n::Int, t::Union{Int, Rational}; io=nothing)
    if io!=nothing
        print(io, "starting eigen decomposition... ")
        flush(io)
    end
    v, P = hamilt |> mat |> Array |> eigen
    if io!=nothing
        println(io, "finished")
    end
    init_vec = init |> statevec
    dt = t//n
    ret = DataFrame(time=[0//1], st_exact=[init])
    for i in 1:n
        t = i * dt
        st = P * Diagonal(exp.(-im * v * t)) * (P' * init_vec)
        push!(ret, (t, ArrayReg(st)))
        if io!=nothing
            print(io, ".")
            flush(io)
        end
    end
    println(io, "")
    flush(io)
    ret
end

function exact_st_evolution(init::AbstractArrayReg, hamilt::AbstractBlock, itr)
    v, P = hamilt |> mat |> Array |> eigen
    init_vec = init |> statevec
    st = P * Diagonal(exp.(-im * v * itr[1])) * (P' * init_vec)
    ret = DataFrame(time=itr[1], st_exact=[ArrayReg(st)])
    for i in itr[2:end]
        st = P * Diagonal(exp.(-im * v * i)) * (P' * init_vec)
        push!(ret, (i, ArrayReg(st)))
    end
    ret
end

function bell_pairs(np, post=I2)
    cir = chain(2np,
        repeat(H, 1:np),
        [cnot(i, i+np) for i in 1:np]...,
        repeat(post, 1:np),
    )
    zero_state(2np) |> cir
end

@memoize function construct_initI(nq)
    IMatrix(2^nq) |> Matrix{ComplexF64} |> matblock
end

function Yao.mat(::Type{T}, gate::YaoUtils.RepeatedRotation) where T
    nq = gate.n
    cb = YaoBlocks.Optimise.to_basictypes(gate)
    initI = construct_initI(nq)
    push!(cb, initI)
    mat(T, cb)
end

# function Yao.mat(::Type{T}, gate::YaoUtils.RepeatedRotation) where T
#     cb = YaoBlocks.Optimise.to_basictypes(gate)
#     mat(T, cb)
# end

function exact_op_evolution(hamilt::AbstractBlock, n::Int, t; io=nothing)
    if io!=nothing
        print(io, "starting eigen decomposition... ")
        flush(io)
    end
    v, P = hamilt |> mat |> Array |> eigen
    if io!=nothing
        println(io, "finished")
    end
    dt = t//n
    ret = DataFrame(time=[0//1], op_exact=[P * Diagonal(exp.(-im * v * 0//1)) * P'])
    for i in 1:n
        t = i * dt
        op = P * Diagonal(exp.(-im * v * t)) * P'
        push!(ret, (t, op))
        if io!=nothing
            print(io, ".")
            flush(io)
        end
    end
    if io!=nothing
        println(io, "finished")
        flush(io)
    end
    ret
end

function tr_mat_CJ(cir::AbstractBlock)
    nq = nqubits(cir)
    bp = bell_pairs(nq)
    focus!(bp, 1:nq)
    st = apply(bp, cir)
    relax!(bp)
    relax!(st)
    bp' * st * 2^nq
end

function mat_CJ(cir::AbstractBlock)
    nq = nqubits(cir)
    st = bell_pairs(nq)
    focus!(st, 1:nq)
    apply!(st, cir)
    relax!(st)
    reshape(st.state, 2^nq, 2^nq) * sqrt(2^nq)
end

end