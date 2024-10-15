module CuYaoUtils

using CuYao
using Main.YaoUtils
export cubell_pairs, cutr_mat_CJ, cumat_CJ

function cubell_pairs(np::Int, post=I2)
    cir = chain(2np,
        repeat(H, 1:np),
        [cnot(i, i+np) for i in 1:np]...,
        repeat(post, 1:np),
    )
    CuYao.cuzero_state(2np) |> cir
end

function cutr_mat_CJ(cir::AbstractBlock)
    nq = nqubits(cir)
    bp = cubell_pairs(nq)
    focus!(bp, 1:nq)
    st = apply(bp, cir)
    relax!(bp)
    relax!(st)
    bp' * st * 2^nq
end

function cumat_CJ(cir::AbstractBlock)
    nq = nqubits(cir)
    st = cubell_pairs(nq)
    focus!(st, 1:nq)
    apply!(st, cir)
    relax!(st)
    reshape(st.state, 2^nq, 2^nq) * sqrt(2^nq)
end

end