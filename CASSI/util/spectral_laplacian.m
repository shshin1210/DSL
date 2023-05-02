function [ out ] = spectral_laplacian( in, boundary )
%SPECTRAL_LAPLACIAN Laplace operator in a spectral domain
%   Laplace operator in a spectral domain.
%   Second derivative between adjacent spectra is computed.
% 
%    [ out ] = SPECTRAL_LAPLACIAN( in, bpimdaru)
%    Inputs
%      in: input spectral image (h,w,lambda)
%      boundary
%        'neumann': Neumann boundary condition (default)
%           e.g. equivalent laplacian matrix = 
%               [ -1 1
%                  1-2 1
%                    ....
%                      1-2 1
%                        ....
%                           1-2 1 
%                             1-1 ]
%        'dirichlet': Dirichlet boundary condition
%           e.g. equivalent laplacian matrix = 
%               [ -2 1
%                  1-2 1
%                    ....
%                      1-2 1
%                        ....
%                           1-2 1 
%                             1-2 ]
%    Outputs
%      out: output (h,w,lambda)

[~,~,d] = size(in);
if nargin < 2
    boundary = 'neumann';
end

switch boundary
    case 'neumann'
        %              f(x+1)-f(x)             f(x+1)+f(x-1)-2f(x)                         f(x-1)-f(x)
        out = cat(3,in(:,:,2)-in(:,:,1),diff(in(:,:,2:d),1,3)-diff(in(:,:,1:(d-1)),1,3),in(:,:,d-1)-in(:,:,d));
    case 'dirichlet'
        %              f(x+1)-2f(x)             f(x+1)+f(x-1)-2f(x)                         f(x-1)-2f(x)
        out = cat(3,in(:,:,2)-2*in(:,:,1),diff(in(:,:,2:d),1,3)-diff(in(:,:,1:(d-1)),1,3),in(:,:,d-1)-2*in(:,:,d));
    otherwise
        error('Options must be neumann or dirichlet.');
end

end