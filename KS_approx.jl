using DifferentialEquations
using ApproxFun, Sundials
using PyPlot

#using Pkg
#Pkg.add("DiffEqOperators")
using DiffEqOperators, LinearAlgebra
begin
	S = Fourier()
	#S = Chebyshev()
	n = 2^8
	x = points(S, n)
	T = ApproxFun.plan_transform(S, n)
	Ti = ApproxFun.plan_itransform(S, n)

	d = 0.2
	nu = 2e-2
	dt = 1e-3
	Nmax =  (3000+1)
	t0 = Nmax * dt

	u₀ = T*(cos.(x).*(sin.(x).+1));
end


function d_dx(S, n, N)
    D = Derivative(S,n)
    return D[1:N,1:N]    
end    

Dx(S,m) = d_dx(S,m,n)
L1 = Dx(S,1)
L2 = Dx(S,2)
L4 = Dx(S,4)

A = DiffEqArrayOperator(-d*Diagonal(L2)-nu*Diagonal(L4))

function f(dû,û,tmp,t)
  # Transform u back to point-space
  mul!(tmp,Ti,û)
  tmp= 0.5.*tmp.*tmp
  mul!(tmp, T, tmp)
  mul!(dû,-L1,tmp)    
  #mul!(dû,T,tmp) # Transform back to Fourier space
end

prob = SplitODEProblem(A, f, u₀, (0.0,t0), similar(u₀));

sol = solve(prob, ETDRK4(), dt=dt)

plot(Ti*sol(0.001))
plot(Ti*sol(3))
#plot(Ti*sol(3))


#
u = zeros(eltype(sol),(n, Nmax))
for it in 1:Nmax
   print(dt*it)
   u[:,it] = Ti*sol(dt*it)
   #println(u[:,it]) 
end
size(u)


IM = PyPlot.imshow(abs.(u))
cb = PyPlot.colorbar(IM, orientation="horizontal") 
#gcf()
