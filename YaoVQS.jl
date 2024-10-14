module YaoVQS

using LinearAlgebra, Yao, DataFrames
using Main.YaoUtils
export vqs_g, vqs_gp, vqs_MV, vqs_real, vqs_MC, vqs_imag, vqs_imag_eig

function vqs_g(init, ansatz, θ)
    cir = if ansatz.blocks[1] isa Union{YaoUtils.RepeatedRotation, YaoUtils.LayeredRepeatedRotation}
        set_params(ansatz, θ)
    else
        dispatch(ansatz, θ)
    end
    g_ansatz = grads(cir)
    @assert length(g_ansatz) == length(θ)
    len = length(θ)
    ret = Vector{typeof(init)}(undef, len)
    Threads.@threads for i in 1:len
        ret[i] = apply(init, g_ansatz[i])
    end
    ret
end

function vqs_gp(init, ansatz, θ)
    gs = vqs_g(init, ansatz, θ)
    len = length(θ)
    cir = if ansatz.blocks[1] isa Union{YaoUtils.RepeatedRotation, YaoUtils.LayeredRepeatedRotation}
        set_params(ansatz, θ)
    else
        dispatch(ansatz, θ)
    end
    st = apply(init, cir)
    ret = Vector{ComplexF64}(undef, len)
    for i in 1:len
        ret[i] = gs[i]' * st
    end
    gs, ret, st
end

function vqs_MV(init, ansatz, θ, hamilt)
    gs, ps, st = vqs_gp(init, ansatz, θ)
    len = length(θ)

    ret_M = Matrix{Float64}(undef, len, len)
    #compute "A" matrix
    for i in 1:len
        for j in i:len
            ret_M[i,j] = ret_M[j,i] = real(gs[i]' * gs[j])
        end
    end

    #compute "M" matrix
    for i in 1:len
        for j in i+1:len
            m = real(ps[i] * ps[j])
            ret_M[i,j] += m
            ret_M[j,i] += m
        end
    end
    for i in 1:len
        ret_M[i,i] += real(ps[i] * ps[i])
    end

    ret_V = Vector{Float64}(undef, len)
    h_st = apply(st, hamilt)
    st_h_st = real(st' * h_st)
    #compute "C" vector
    for i in 1:len
        ret_V[i] = imag(gs[i]' * h_st)
    end

    #compute "V" vector
    for i in 1:len
        ret_V[i] -= imag(ps[i])
    end
    
    ret_M, ret_V
end

function vqs_MC(init::AbstractRegister, ansatz::AbstractBlock, θ, hamilt::AbstractBlock)
    gs, ps, st = vqs_gp(init, ansatz, θ)
    len = nparameters(ansatz)

    ret_M = Matrix{Float64}(undef, len, len)
    #compute "A" matrix
    for i in 1:len
        for j in i:len
            ret_M[i,j] = ret_M[j,i] = real(gs[i]' * gs[j])
        end
    end

    #compute "M" matrix
    for i in 1:len
        for j in i+1:len
            m = real(ps[i] * ps[j])
            ret_M[i,j] += m
            ret_M[j,i] += m
        end
    end
    for i in 1:len
        ret_M[i,i] += real(ps[i] * ps[i])
    end

    ret_C = Vector{ComplexF64}(undef, len)
    h_st = apply(st, hamilt)
    st_h_st = real(st' * h_st)
    #compute "C" vector
    for i in 1:len
        ret_C[i] = gs[i]' * h_st
    end
    
    ret_M, ret_C
end

function vqs_real(init, ansatz, hamilt, n, t, init_params, diag, integration, callback)
    dt = t//n
    ret = DataFrame(time=[0//n], θ=[init_params])
    callback(0, 0//n, init_params)
    θ = init_params
    t = 0//n
    for i in 1:n
        t += dt
        dθ = if integration == :euler
            M,V = vqs_MV(init, ansatz, θ, hamilt)
            (M + I*diag) \ V
        elseif integration == :rk4
            M,V = vqs_MV(init, ansatz, θ, hamilt)
            dθ1 = (M + I*diag) \ V
            M,V = vqs_MV(init, ansatz, θ + dθ1*dt/2, hamilt)
            dθ2 = (M + I*diag) \ V
            M,V = vqs_MV(init, ansatz, θ + dθ2*dt/2, hamilt)
            dθ3 = (M + I*diag) \ V
            M,V = vqs_MV(init, ansatz, θ + dθ3*dt, hamilt)
            dθ4 = (M + I*diag) \ V
            (dθ1 + 2dθ2 + 2dθ3 + dθ4)/6
        else
            error("integration $(integration) is not supported")
        end
        θ += dθ * dt
        ps = (t, θ)
        push!(ret, ps)
        if callback(i, ps[1], ps[2])
            break
        end
    end
    ret
end

function vqs_imag(init, ansatz, hamilt, n, t, init_params, diag, integration, callback)
    dt = t//n
    ret = DataFrame(time=[0//n], θ=[init_params])
    callback(0, 0//n, init_params)
    θ = init_params
    t = 0//n
    for i in 1:n
        t += dt
        dθ = if integration == :euler
            M,C = vqs_MC(init, ansatz, θ, hamilt)
            -(M + I*diag) \ real(C)
        elseif integration == :rk4
            M,C = vqs_MC(init, ansatz, θ, hamilt)
            dθ1 = (M + I*diag) \ real(C)
            M,C = vqs_MC(init, ansatz, θ + dθ1*dt/2, hamilt)
            dθ2 = (M + I*diag) \ real(C)
            M,C = vqs_MC(init, ansatz, θ + dθ2*dt/2, hamilt)
            dθ3 = (M + I*diag) \ real(C)
            M,C = vqs_MC(init, ansatz, θ + dθ3*dt, hamilt)
            dθ4 = (M + I*diag) \ real(C)
            -(dθ1 + 2dθ2 + 2dθ3 + dθ4)/6
        else
            error("integration $(integration) is not supported")
        end
        θ += dθ * dt
        ps = (t, θ)
        push!(ret, ps)
        if callback(i, ps[1], ps[2])
            break
        end
    end
    ret
end

function vqs_imag_eig(init, ansatz, hamilt, n, t, init_params, diag, integration, callback, tol)
    dt = t//n
    energy = expect(hamilt, apply(init, set_params(ansatz, init_params)))
    ret = DataFrame(time=[0//n], energy=[energy], θ=[init_params])
    callback(0, 0//n, init_params)
    θ = init_params
    t = 0//n
    for i in 1:n
        t += dt
        dθ = if integration == :euler
            M,C = vqs_MC(init, ansatz, θ, hamilt)
            -(M + I*diag) \ real(C)
        elseif integration == :rk4
            M,C = vqs_MC(init, ansatz, θ, hamilt)
            dθ1 = (M + I*diag) \ real(C)
            M,C = vqs_MC(init, ansatz, θ + dθ1*dt/2, hamilt)
            dθ2 = (M + I*diag) \ real(C)
            M,C = vqs_MC(init, ansatz, θ + dθ2*dt/2, hamilt)
            dθ3 = (M + I*diag) \ real(C)
            M,C = vqs_MC(init, ansatz, θ + dθ3*dt, hamilt)
            dθ4 = (M + I*diag) \ real(C)
            -(dθ1 + 2dθ2 + 2dθ3 + dθ4)/6
        else
            error("integration $(integration) is not supported")
        end
        θ += dθ * dt
        energy = expect(hamilt, apply(init, set_params(ansatz, θ)))
        ps = (t, energy, θ)
        push!(ret, ps)
        if callback(i, ps[1], ps[2])
            break
        end
        if abs(ret[end-1, :energy] - ret[end, :energy]) < tol
            break
        end
    end
    ret
end

end