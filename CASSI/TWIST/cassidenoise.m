%============================================================================
%
% Codename: Bird Scanner (Yale Computer Graphics Group)
%
% Copyright (C) 2011-12 Min H. Kim. All rights reserved.
%
% GNU General Public License Usage
% Alternatively, this file may be used under the terms of the GNU General
% Public License version 3.0 as published by the Free Software Foundation
% and appearing in the file LICENSE.GPL included in the packaging of this
% file. Please review the following information to ensure the GNU General
% Public License version 3.0 requirements will be met:
% http://www.gnu.org/copyleft/gpl.html.
%
%============================================================================
% function [imgdenoise]=cassidenoise(imgnoise,tau,iter)
function [imgdenoise]=cassidenoise(imgnoise,tau,iter)
%{
    f = double(imread('cameraman.tif'))/255;
    f = f + randn(size(f))*16/255;
    u = cassidenoise(f,0.1,4);
    subplot(1,2,1); imshow(f); title Input
    subplot(1,2,2); imshow(u); title Denoised
%}
% [Min] imgnoise looks like ATy (w*h*spectral)
% This function is crucial for Iterative process and computing redisual% tau=.4;
% iter=4;
% imgnoise=rand(400,400,37);
% tic, [imgdenoise]=mycalltoTVnewMatrix2cpu(imgnoise,tau,iter); toc

%=================== Hard coded for the target dataset =================
R = 580;
C = 890;
% R = 580/2;
% C = 890/2;
imgnoise = reshape(imgnoise, R, C, []);
%======================================================================

x = imgnoise(:,:,1);
[uy ux] = size(x);
uz=size(imgnoise,3);

zv=zeros(1,size(x,2),'single');
zh=zeros(size(x,1),1,'single');
pn=zeros(uy*ux,2,'single');

vect = @(x) x(:); % vecterize
grad = @(x) [vect(dh(x,zh)) vect(dv(x,zv))] ; % two column vectors of horizontal and vertical gradients
div = @(x) dht(reshape(x(:,1),uy,ux),zh)+dvt(reshape(x(:,2),uy,ux),zv); % this is divergence (sum of second order derivatives)

imgdenoise=zeros(uy,ux,uz,'single');
for i = 1:uz
    x = imgnoise(:,:,i);
    
    imgdenoise(:,:,i) = x - projk(x,tau/2,grad,div,iter,pn);
end

%===================Vectorize: Hard coded for the target dataset ======
imgdenoise = imgdenoise(:);
%======================================================================

end

% Subfunctions:
% Proj function:
% [Min] it is not clear. what is it doing?
function u=projk(g,lambda,grad,div,niter,pn)
tau=0.25;
for i=1:niter
    S=grad(-div(pn)-g./lambda);
    pn=(pn+tau*S)./(1+tau*modulo(S));
end
u=-lambda.*div(pn);
end

function R=modulo(x)
R=sqrt(sum(x.^2,2));
R=[R R];
end

% Derivatives:
function y = dh(x,zh)
x = [x zh] - [zh x];
x(:,end) = x(:,end)+x(:,1);
y = x(:,2:end);
end

function y = dht(x,zh)
x = [zh x] - [x zh]; % calculate gradient (horizontal)
x(:,1) = x(:,end)+x(:,1);
y = x(:,1:end-1);
end

function y = dv(x,zv)
x = [x; zv]- [zv; x];
x(end,:) = x(end,:)+x(1,:);
y = x(2:end,:);
end

function y = dvt(x,zv)
x = [zv; x]-[x; zv];
x(1,:) = x(end,:)+x(1,:);
y = x(1:end-1,:);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Original version:
function u = tvdenoise(f,lambda,iters)
%TVDENOISE  Total variation grayscale and color image denoising
%   u = TVDENOISE(f,lambda) denoises the input image f.  The smaller
%   the parameter lambda, the stronger the denoising.
%
%   The output u approximately minimizes the Rudin-Osher-Fatemi (ROF)
%   denoising model
%
%       Min  TV(u) + lambda/2 || f - u ||^2_2,
%        u
%
%   where TV(u) is the total variation of u.  If f is a color image (or any
%   array where size(f,3) > 1), the vectorial TV model is used,
%
%       Min  VTV(u) + lambda/2 || f - u ||^2_2.
%        u
%
%   TVDENOISE(...,Tol) specifies the stopping tolerance (default 1e-2).
%
%   The minimization is solved using Chambolle's method,
%      A. Chambolle, "An Algorithm for Total Variation Minimization and
%      Applications," J. Math. Imaging and Vision 20 (1-2): 89-97, 2004.
%   When f is a color image, the minimization is solved by a generalization
%   of Chambolle's method,
%      X. Bresson and T.F. Chan,  "Fast Minimization of the Vectorial Total
%      Variation Norm and Applications to Color Image Processing", UCLA CAM
%      Report 07-25.
%
%   Example:
%   f = double(imread('barbara-color.png'))/255;
%   f = f + randn(size(f))*16/255;
%   u = tvdenoise(f,12);
%   subplot(1,2,1); imshow(f); title Input
%   subplot(1,2,2); imshow(u); title Denoised

% Pascal Getreuer 2007-2008
%  Modified by Jose Bioucas-Dias  & Mario Figueiredo 2010 
%  (stopping rule: iters)
%   

if nargin < 3
    Tol = 1e-2;
end

if lambda < 0
    error('Parameter lambda must be nonnegative.');
end

dt = 0.25;

N = size(f);
id = [2:N(1),N(1)];
iu = [1,1:N(1)-1];
ir = [2:N(2),N(2)];
il = [1,1:N(2)-1];
p1 = zeros(size(f));
p2 = zeros(size(f));
divp = zeros(size(f));
lastdivp = ones(size(f));

if length(N) == 2           % TV denoising
    %while norm(divp(:) - lastdivp(:),inf) > Tol
    for i=1:iters
        lastdivp = divp;
        z = divp - f*lambda;
        z1 = z(:,ir) - z;
        z2 = z(id,:) - z;
        denom = 1 + dt*sqrt(z1.^2 + z2.^2);
        p1 = (p1 + dt*z1)./denom;
        p2 = (p2 + dt*z2)./denom;
        divp = p1 - p1(:,il) + p2 - p2(iu,:);
    end
elseif length(N) == 3       % Vectorial TV denoising
    repchannel = ones(N(3),1);

    %while norm(divp(:) - lastdivp(:),inf) > Tol
    for i=1:iters
        lastdivp = divp;
        z = divp - f*lambda;
        z1 = z(:,ir,:) - z;
        z2 = z(id,:,:) - z;
        denom = 1 + dt*sqrt(sum(z1.^2 + z2.^2,3));
        denom = denom(:,:,repchannel);
        p1 = (p1 + dt*z1)./denom;
        p2 = (p2 + dt*z2)./denom;
        divp = p1 - p1(:,il,:) + p2 - p2(iu,:,:);
    end
end

u = f - divp/lambda;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the old Phi denoising function (of the old CASSI)
% This is the virtually same code as the latest version.
function y=MyTVphi(x,Nvx,Nvy,Nvz,weights)
    if (nargin < 5), weights = 1;  end

    X=reshape(x,Nvx,Nvy,Nvz);

    [y,dif]=MyTVnorm(X,weights);
end

function [y,dif]=MyTVnorm(x,weights)
    if (nargin < 2), weights = 1;  end

    TV=MyTV3D_conv(x,weights);

    dif=sqrt(sum(abs(TV).^2,4));

    y=sum(dif(:));
end

function TV=MyTV3D_conv(x,weights)
    if (nargin < 2), weights = 1;  end

    [nx,ny,nz]=size(x);
    TV=zeros(nx,ny,nz,3);

    TV(:,:,:,1)=circshift(x,[-1 0 0])-x;
    TV(nx,:,:,1)=0.0;

    TV(:,:,:,2)=circshift(x,[0 -1 0])-x;
    TV(:,ny,:,2)=0.0;

    TV(:,:,:,3)=circshift(x,[0 0 -1])-x;
    TV(:,:,nz,3)=0.0;

    [nx_w ny_w nz_w n_w]=size(weights);
    if([nx_w ny_w nz_w n_w]==[1 3 1 1])
        TV(:,:,:,1)=TV(:,:,:,1).*weights(1);
        TV(:,:,:,2)=TV(:,:,:,2).*weights(2);
        TV(:,:,:,3)=TV(:,:,:,3).*weights(3);
    else
        TV=TV.*weights;
    end;
end

function [y,dif]=MyTV3Dnorm_conv(x)
    TV=MyTV3D_conv(x);
    dif=sqrt(TV(:,:,:,1).^2+TV(:,:,:,2).^2+TV(:,:,:,3).^2);
    % dif=abs(TV(:,:,:,1))+abs(TV(:,:,:,2))+abs(TV(:,:,:,3));
    y=sum(dif(:));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the old Psi denoising function (of the old CASSI)
% This is the virtually same code as the latest version.
function y=MyTVpsi(x,th,tau,iter,Nvx,Nvy,Nvz,weights)
    if (nargin < 8), weights = 1;  end

    X=reshape(x,Nvx,Nvy,Nvz);

    y=X-MyProjectionTV(X,tau,th*0.5,iter,weights);

    % y=reshape(Y,Nvx*Nvy*Nvz,1);
end

function p=MyProjectionTV(g,tau,lam,iter,weights)
    if (nargin < 5), weights = 1;  end

    [nx,ny,nz]=size(g);
    pn=zeros(nx,ny,nz,3);
    div_pn=zeros(nx,ny,nz);
    b=pn;

    for i=1:iter

        a=MyTV3D_conv(div_pn-g./lam,weights);

        b(:,:,:,1)=sqrt(a(:,:,:,1).^2+a(:,:,:,2).^2+a(:,:,:,3).^2);
        b(:,:,:,2)=b(:,:,:,1);
        b(:,:,:,3)=b(:,:,:,1);
        pn=(pn+tau.*a)./(1.0+tau.*b);

    %     b(:,:,:,1)=a(:,:,:,1).^2+a(:,:,:,2).^2+a(:,:,:,3).^2;
    %     b(:,:,:,2)=b(:,:,:,1);
    %     b(:,:,:,3)=b(:,:,:,1);    
    %     pn=(pn+tau.*a)./sqrt(1.0+(tau.^2).*b);

        div_pn=MyDiv3D(pn);
    end;

    p=lam.*MyDiv3D(pn);
end

function y=MyDiv3D(TV)
    n=size(TV);

    x_shift=circshift(TV(:,:,:,1),[1 0 0 0]);
    yx=TV(:,:,:,1)-x_shift;
    yx(1,:,:)=TV(1,:,:,1);
    yx(n(1),:,:)=-x_shift(n(1),:,:);

    y_shift=circshift(TV(:,:,:,2),[0 1 0 0]);
    yy=TV(:,:,:,2)-y_shift;
    yy(:,1,:)=TV(:,1,:,2);
    yy(:,n(2),:)=-y_shift(:,n(2),:);

    z_shift=circshift(TV(:,:,:,3),[0 0 1 0]);
    yz=TV(:,:,:,3)-z_shift;
    yz(:,:,1)=TV(:,:,1,3);
    yz(:,:,n(3))=-z_shift(:,:,n(3));


    y=yx+yy+yz;
end