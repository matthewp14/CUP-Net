function u=denoise_func_new(x,th,Nx,Ny,Nz,tv_iters)

x=reshape(x,Nx,Ny,Nz);
u=tvdenoise(x,2/th,tv_iters);
u = u(:);