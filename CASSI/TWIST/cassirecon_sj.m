
function cassirecon_sj(root_path, scene_path, crect, wvls2b, flipped)


%root_path = '../data/20151221_colorchecker2/';

% Path for captured images
if nargin < 2
    scene_path = [root_path '/scene_warped/'];
end
scene_files = dir([scene_path '/*.png']);

% Path for coded masks
coded_mask_path = [root_path '/cells_coded_warped/'];
coded_mask_files = dir([coded_mask_path '/*.png']);

% Dimensions
im = imread([scene_path '/' scene_files(1).name]);

% n1 = size(im,1);
% n2 = size(im,2);
% m = 28;
% nt = 1;

%% Select Crop Area
if nargin < 3
    [ ~, crect ] = selectregion( im );
end
n1 = crect(4) + 1;
n2 = crect(3) + 1;

if nargin < 4
    wvls2b = [420:10:700];
end

m = size(wvls2b, 2);
nt = size(scene_files, 1); % number of views
solver = 'twist';

if nargin < 4
   flipped = [1 5]; 
end

clear( 'X' );

%% Pack Captured Images
y = zeros(n1, n2, nt);
for i=1:nt
    file = scene_files(i);
    scene = im2double(imread([scene_path file.name]));
    
    % Crop the scene
    scene = scene(crect(2):crect(2)+crect(4) , crect(1):crect(1)+crect(3),: );
    y(:,:,i) = scene;
end

y = y.*(y>=0);
y = y(:);

%% Dispersion
disperse_csv = csvread('data/dispersed.csv');
wvls = disperse_csv(:, 1);
disperse = disperse_csv(:, 2);
fittedmodel = fit(wvls, disperse, 'power2');
ref_disperse = fittedmodel(636);
disperse = fittedmodel(wvls2b) - ref_disperse;

if nt > 1 && exist([root_path '/homographys.mat'])
    load([root_path '/homographys.mat']);

    for i=1:nt
        disperses(i,:) = int8(disperse ./ abs(homographys{i}.tform.T(1, 1) / homographys{i}.tform.T(3, 3)));
    end
else
    for i=1:nt
        disperses(i,:) = int8(disperse);
    end
end


%% Construct Projection & Coding Matrix
Cu = zeros(n1, n2, m, nt);
for i=1:nt
    file = coded_mask_files(i);
    coded_mask = im2double(imread([coded_mask_path file.name]));
%     coded_mask = stretch_hist(coded_mask);
    
%     coded_mask = stretch_hist(coded_mask); % [MK] hist stretch -> FINAL
%    coded_mask = stretch_hist(coded_mask,3,99); % [MK] these parameters are not good
%    figure; imshow(coded_mask);
%    coded_mask = histeq(coded_mask); % [MK] this is very bed
%    coded_mask = imsharpen(coded_mask,'Radius',2,'Amount',1); % [MK] unsharpen masking filter -> bad!
    
    coded_mask = coded_mask(crect(2):crect(2)+crect(4) , crect(1):crect(1)+crect(3),: );
    for wave_number=1:m
        Cu(:,:,wave_number,i) = coded_mask;
    end
end


%% Prepare Parameters

tau = 0.15; % smoothing factor parameters
tv_iter = 20; % numger of iteration in a single denoise (for tvdenoise.m)
iterA = 15; % max iteration for TwIST
tolA = 1e-4; % Iteration stop criteria in TwIST


nt = 1;
y;
A = @(f) R2(f,n1,n2,m,Cu,nt,disperses,flipped);
AT = @(y) RT2(y,n1,n2,m,Cu,nt,disperses,flipped);
Psi = @(x,th) cassidenoise(x,th,tv_iter);
Phi = @(x) TVnorm3D(x);

switch lower(solver)
    case 'twist'
        % Reconstruct using TwIST
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
end




%%---------------------------------------------------------------------------------

% x_recon = shiftCube(x_recon,-3);

%%---------------------------------------------------------------------------------

%% Write results
result_path = [scene_path '/result/'];
mkdir(result_path);
mkdir([result_path '/bands_raw/']);
mkdir([result_path '/bands_norm/']);
mkdir([result_path '/bands_enhanced/']);

n = size(x_recon, 3);
for i=1:n
    imwrite(x_recon(:,:,i), sprintf('%s/band_%02d.png', [result_path '/bands_raw/'], i));
end

n = size(x_recon, 3);
for i=1:n
    imwrite(x_recon(:,:,i) * 8, sprintf('%s/band_%02d.png', [result_path '/bands_enhanced/'], i));
end

n = size(x_recon, 3);
for i=1:n
    imwrite(mat2gray(x_recon(:,:,i)), sprintf('%s/band_%02d.png', [result_path '/bands_norm/'], i));
end

q = A(x_recon);
q = reshape(q, n1, n2, nt);
n = size(q, 3);
for i=1:n
    imwrite(q(:,:,i), sprintf('%s/reconstruct_%02d.png', result_path, i));
end

output_path = [result_path sprintf('/_view%d_solver%s.mat', nt, upper(solver))];
save(output_path, 'x_recon', 'wvls2b');

sRGB = R2sRGB(wvls2b,x_recon,0, 0, 0);
imwrite(sRGB, [result_path '/rgb.png']);

%cassidisplay(0, 0, output_path);

end

function y = TVnorm3D(x)
% this is phi function (this produces the summation of the magnitudes of gradients)
% TVnonmspectralimging --> one constant
m=size(x,3);

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


function y=R2(f,n1,n2,m,Cs,nt,disperses,flipped) % y = Ax (h*w*snap)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Min] Here multidimensional dot product was impremented 
%       as elementary product plus sum 
%       1: h, 2: w, 3: spectral channels, 4: aperture snaps
% Elementary product
% Sum up the 3rd dimensions in each seperate image:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f=reshape(f,[n1,n2,m]);
gp=repmat(f,[1 1 1 nt]).*Cs; % 4D * 4D


% Disperse
for i = 1:m
    gp_1channels = gp( :, :, i, : );
    
    for j = flipped
        gp( :, :, i, j ) = circshift( gp_1channels(:,:,1,j), [ 0, -disperses(j, i), 0 ] );
    end
    
    other = 1:nt;
    other(flipped) = [];
    for j = other
        gp( :, :, i, j ) = circshift( gp_1channels(:,:,1,j), [ 0, disperses(j, i), 0 ] );
    end
end

% Projection
y=sum(gp,3); % 4D -> 3D
y=y(:); % vectorize
end

function f=RT2(y,n1,n2,m,Cs,nt,disperses,flipped) % f = ATy (h*w*spec)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Min] Here multidimensional dot product was impremented 
%       as elementary product plus sum 
%       1: h, 2: w, 3: spectral channels, 4: aperture snaps
% Elementary product
% Sum up on the 4th dimension:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y=reshape(y,[n1,n2,1,nt]);
yp=repmat(y,[1,1,m,1]);


% Disperse
for i = 1:int32( m )
    yp_1channels = yp( :, :, i, : );
    
    for j = flipped
        yp( :, :, i, j ) = circshift( yp_1channels(:,:,1,j), [ 0, disperses(j, i), 0 ] );
    end
    
    other = 1:nt;
    other(flipped) = [];
    for j = other
        yp( :, :, i, j ) = circshift( yp_1channels(:,:,1,j), [ 0, -disperses(j, i), 0 ] );
    end
end

yp=yp.*Cs; % 4D * 4D
f=sum(yp,4); % DUKE: 4D -> 3D
end

