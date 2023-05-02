function output = recon_spectrum_CASSI_single(data_path, result_path)

%% parameters
if nargin < 1
    data_path = './data/balloons_ms/';
    result_path = './results/';
end
if ~exist(result_path, 'dir')
    mkdir(result_path);
end
addpath('util\');
% addpath('recon\');
addpath('TWIST\');
% addpath('model\');

% depth = hdf5read(data_path, 'depth');
% depth_n = hdf5read(data_path, 'depth_n');


msim_list = dir([data_path '*.png']);
nChannels = numel(msim_list);
im0 = imread([data_path msim_list(1).name]);
[N,M] = size(im0);
% prepare multi-spectral images
hs = [];
for i = 1:nChannels
    im = im2double(imread([data_path msim_list(i).name]));
    hs = cat(3, hs, im);
end

% hs = permute(h5read(data_path, '/hs'), [3 2 1]);
% wvls2b = h5read(data_path, '/wvls2b');
wvls2b = 400:10:700;
% code = im2double(imread('code.png'));
code = rand(N, M);
code(:)= code>0.5;
% wvls2b = wvls2b(:, 1:25);
% hs = hs(:, :, 1:25);

%% rescale
% scaler = 0.5;
% hs = imresize(hs, scaler, 'bilinear');
% depth = imresize(depth, scaler, 'nearest');
gt = hs;

%%
[n1, n2, m] = size(hs);
code = code(1:n1, 1:n2);
Cu = zeros(n1, n2, m);
for i=1:m
    Cu(:, :, i) = imtranslate(code, [i-13, 0]);
end

y = R(gt, n1, n2, m, Cu) / m;

% QPhi = @(x)compute_A_mul_any_coded_mask(x, shifted_ind, spec2grey, 1, coded_mask);
% QPhi_T= @(y)compute_A_mul_any_coded_mask(y, shifted_ind, spec2grey, 2, coded_mask);
% tau = 3e-1; %0.15; % smoothing factor parameters
% tv_iter = 20; % numger of iteration in a single denoise (for tvdenoise.m)
% iterA = 40; % max iteration for TwIST
% tolA = 1e-5; % Iteration stop criteria in TwIST

tau = 3e-1; %0.05;
tv_iter = 40; % numger of iteration in a single denoise (for tvdenoise.m)
iterA = 200; % max iteration for TwIST
tolA = 1e-4; % Iteration stop criteria in TwIST

A = @(f) R(f,n1,n2,m,Cu);
AT = @(y) RT(y,n1,n2,m,Cu);
% Psi = @(x,th) cassidenoise(x,th,tv_iter,n1,n2,m);
Psi = @(x,th) cassidenoise(x,th,tv_iter);
Phi = @(x) TVnorm3D(x,n1,n2,m);

[x_recon,dummy,obj_twist,times_twist,dummy,mse_twist] = TwIST( ...
    y,A,tau,...
    'AT', AT, ...
    'Psi', Psi, ...
    'Phi',Phi, ...
    'Initialization',2,...
    'Monotone',1,...
    'StopCriterion',1,...
    'MaxIterA',iterA,...
    'ToleranceA',tolA,...
    'Debias',0,...
    'Verbose', 1);

x_recon = reshape(x_recon, [n1, n2, m]);
save([result_path, 'cassi.mat'], 'x_recon', 'wvls2b', 'gt');
end

function y = TVnorm3D(x,n1,n2,m)
% this is phi function (this produces the summation of the magnitudes of gradients)
% TVnonmspectralimging --> one constant
x = reshape(x, [n1, n2, m]);

shift1 = circshift(x, [-1 0 0]);
shift2 = circshift(x, [0 -1 0]);

sub_x = x(1:end-1,1:end-1,:);
shift1 = shift1(1:end-1,1:end-1,:);
shift2 = shift2(1:end-1,1:end-1,:);

% L-2 norm [SJ]
% diff1 = (sub_x - shift1) .^ 2;
% diff2 = (sub_x - shift2) .^ 2;
% y = diff1 + diff2;
% y = sqrt(y);
% y = sum(y(:));

% L-1 norm [MK] --> slightly better!
diff1 = abs(sub_x - shift1);
diff2 = abs(sub_x - shift2);

y = diff1 + diff2;
y = sum(y(:));

end

function y=R(f,n1,n2,m,Cs) % y = Ax (h*w*snap)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Min] Here multidimensional dot product was impremented 
%       as elementary product plus sum 
%       1: h, 2: w, 3: spectral channels, 4: aperture snaps
% Elementary product
% Sum up the 3rd dimensions in each seperate image:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f=reshape(f,[n1,n2,m]); % 1D -> 3D
%figure(fig);
gp=f.*Cs; % 3D * 3D

% Projection
y=sum(gp,3); % 3D -> 2D
%imshow(y);
y=y(:); % vectorize
end

function f=RT(y,n1,n2,m,Cs) % f = ATy (h*w*spec)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Min] Here multidimensional dot product was impremented 
%       as elementary product plus sum 
%       1: h, 2: w, 3: spectral channels, 4: aperture snaps
% Elementary product
% Sum up on the 4th dimension:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y=reshape(y,[n1,n2,1]); % 1D -> 2D
%figure(fig);
yp=repmat(y,[1,1,m]);

yp=yp.*Cs; % 3D * 3D
f=yp;
f = f(:); % vectorize
end
