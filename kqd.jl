"""
kqd
"""
function kqd(hamilt::AbstractBlock, init::ArrayReg, dt::Real, d::Int)
    psi = [apply(init, TimeEvolution(hamilt, t)) for t in range(start=0, step=dt, length=d)]
    H = Matrix{ComplexF64}(undef, d, d)
    S = Matrix{ComplexF64}(undef, d, d)
    for j in 1:d, k in 1:d
        H[j,k] = psi[j]' * apply(psi[k], hamilt)
        S[j,k] = psi[j]' * psi[k]
    end
    vals, vecs = eigen(H,S)
    (;val = vals[1], vec = vecs[:,1])
end

function kqd(hamilts::AbstractVector{Tb}, init::ArrayReg, dt::Real, d::Int) where {Tb<:AbstractBlock}
    nq = nqubits(init)
    te = chain(nq, [TimeEvolution(hamilts[i], dt) for i in 1:length(hamilts)]...)
    psi = [apply(init, te^i) for i in 0:d-1]
    H = Matrix{ComplexF64}(undef, d, d)
    S = Matrix{ComplexF64}(undef, d, d)
    for j in 1:d, k in 1:d
        H[j,k] = psi[j]' * apply(psi[k], hamilt)
        S[j,k] = psi[j]' * psi[k]
    end
    vals, vecs = eigen(H,S)
    (;val = vals[1], vec = vecs[:,1])
end