

function make_simulation_results_cassi( scene_name,...
                                        input_dir,...
                                        input_code_dir, ...
                                        mask_filename_ordering, ...
                                        n_multi_samples,...
                                        input_dispersion_csv,...
                                        output_dir, ...
                                        flipped)
%% If output_dir doest exist
output_dir = fullfile( output_dir, scene_name ); 
if ~exist( output_dir, 'dir' )
   mkdir( output_dir ); 
end

%% Load Multi Sampled CASSI Images
scene_files = cell( n_multi_samples, 1 );
for i = 1:n_multi_samples
    filename = sprintf( '%s_%02d.png', scene_name, i );
    scene_files{ i } = fullfile( input_dir, filename );
end

%% Load Coded Masks
coded_mask_path = 'data/coded/';
coded_mask_files = cell( n_multi_samples, 1 );
for i = 1:n_multi_samples
    filename = sprintf( 'cell_%s.png', mask_filename_ordering{ i } );
    coded_mask_files{ i } = fullfile( input_code_dir, filename );
end


%% Load Dispersion
disperse_csv = csvread( input_dispersion_csv );
wvls = disperse_csv(:, 1);
disperse = disperse_csv(:, 2);
fittedmodel = fit(wvls, disperse, 'power2');
ref_disperse = fittedmodel( 636 );

spectrum_start_idx = 6;                 % 450 nm
spectrum_end_idx = 30;                  % 690 nm
n_spectums = spectrum_end_idx - spectrum_start_idx + 1;
wvls2b...
    = ( 400 + ( spectrum_start_idx - 1 )*10 ):10:( 400 + ( spectrum_end_idx - 1 )*10 );
disperse = int8( fittedmodel( wvls2b ) - ref_disperse );

% Dimensions
im = imread( scene_files{ i } );
n1 = size(im,1);
n2 = size(im,2);
m = length(wvls2b);
nt = n_multi_samples;

%% Pack Captured Images
y = zeros( n1, n2, nt );
for i=1:nt
    file =  scene_files{ i };
    scene = im2double(imread( file ));
    y( :, :, i ) = scene;
end
% file = scene_files(3);
% scene = im2double(imread([scene_path file.name]));
% y(:,:,1) = scene;

y = y.*(y>=0);
y = y(:);


%% Construct Projection & Coding Matrix
Cu = zeros( n1, n2, m, nt );
for i = 1:nt
    file = coded_mask_files{ i };
    coded_mask = im2double(imread( file ));
%     figure, imshow( coded_mask ), title( 'before' );
    
    coded_mask = stretch_hist(coded_mask); % [MK] hist stretch -> FINAL
%     figure, imshow( coded_mask ), title( 'after' );
%    coded_mask = stretch_hist(coded_mask,3,99); % [MK] these parameters are not good
%    figure; imshow(coded_mask);
%    coded_mask = histeq(coded_mask); % [MK] this is very bed
%    coded_mask = imsharpen(coded_mask,'Radius',2,'Amount',1); % [MK] unsharpen masking filter -> bad!
    
    for wave_number=1:m
        Cu(:,:,wave_number,i) = coded_mask;
    end
end
% file = coded_mask_files(3);
% coded_mask = im2double(imread([coded_mask_path file.name]));
% 
% for wave_number=1:m
%     Cu(:,:,wave_number,1) = coded_mask;
% end


%% Prepare Parameters

tau = 0.15; % smoothing factor parameters
tv_iter = 10; % numger of iteration in a single denoise (for tvdenoise.m)
iterA = 50; % max iteration for TwIST
% iterA = 1; % max iteration for TwIST
tolA = 1e-8; % Iteration stop criteria in TwIST

A = @(f) R2(f,n1,n2,m,Cu,nt,disperse, flipped);
AT = @(y) RT2(y,n1,n2,m,Cu,nt,disperse, flipped);
Psi = @(x,th) cassidenoise(x,th,tv_iter);
Phi = @(x) TVnorm3D(x);

%% Reconstruct using TwIST
[x_twist,dummy,obj_twist,times_twist,dummy,mse_twist] = TwIST( ...
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

%temp = shiftCube(x_twist_orig);
%x_twist_orig = temp; 


%% Write results
output_dir_gray = fullfile( output_dir, 'gray' );
output_dir_rgb = fullfile( output_dir, 'rgb' );
output_dir_verification = fullfile( output_dir, 'verification' );
if ~exist( output_dir_gray, 'dir' )
    mkdir( output_dir_gray );
end

if ~exist( output_dir_rgb, 'dir' )
    mkdir( output_dir_rgb );
end

if ~exist( output_dir_verification, 'dir' )
    mkdir( output_dir_verification );
end

n = size(x_twist, 3);
for i=1:n
    wvls = wvls2b(i);
    im_1channel = x_twist( :, :, i);
    
    imwrite(im_1channel * 8, sprintf( '%s/%dnm.png', output_dir_gray, wvls));
    
    im_rgb = zeros(size(im_1channel));
    sRGB = spectrumRGB( wvls );
    im_rgb(:,:,1) = im_1channel .* sRGB(1);
    im_rgb(:,:,2) = im_1channel .* sRGB(2);
    im_rgb(:,:,3) = im_1channel .* sRGB(3);
    imwrite(im_rgb * 8, sprintf( '%s/%d_nm.png', output_dir_rgb, wvls));
end

%{
n = size(x_twist, 3);
for i=1:n
    imwrite(mat2gray(x_twist(:,:,i)), sprintf('result/norm_band_%02d.png', i));
end
%}

q = A(x_twist);
q = reshape(q, n1, n2, nt);
n = size(q, 3);
for i=1:n
    imwrite( q(:,:,i), sprintf('%s/reconstruct_%02d.png', output_dir_verification, i));
end

mat_file = sprintf('%s/%s_recon_spectrum.mat', output_dir, scene_name);
save( mat_file,'x_twist', 'wvls2b' );
[refwhite,sRGB] = cassidisplay_for_simulation( 0, 0, mat_file );
imwrite( sRGB, sprintf('%s/%s_recon_sRGB.png', output_dir, scene_name ) );

end

function y = TVnorm3D(x)
% this is phi function (this produces the summation of the magnitudes of gradients)
% TVnonmspectralimging --> one constant
m = size( x, 3 );

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


function y=R2( f, n1, n2, m, Cs, nt, disperse, flipped ) % y = Ax (h*w*snap)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Min] Here multidimensional dot product was impremented 
%       as elementary product plus sum 
%       1: h, 2: w, 3: spectral channels, 4: aperture snaps
% Elementary product
% Sum up the 3rd dimensions in each seperate image:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f = reshape(f,[n1,n2,m]);
gp=repmat(f,[1 1 1 nt]).*Cs; % 4D * 4D


% Disperse
for i = 1:m
    gp_1channels = gp( :, :, i, : );
    
    
    gp( :, :, i, flipped ) = circshift( gp_1channels(:,:,1,flipped), [ 0, -disperse( i ), 0 ] );
    gp( :, :, i, setdiff(1:nt, flipped) ) = circshift( gp_1channels(:,:,1,setdiff(1:nt, flipped)), [ 0, disperse( i ), 0 ] );
    %gp( :, :, i, : ) = circshift( gp_1channels, [ 0, disperse( i ), 0, 0 ] );
end

% Projection
y=sum(gp,3); % 4D -> 3D
y=y(:); % vectorize
end

function f = RT2( y, n1, n2, m, Cs, nt, disperse, flipped ) % f = ATy (h*w*spec)
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
    
    
    yp( :, :, i, flipped ) = circshift( yp_1channels(:,:,1,flipped), [ 0, disperse( i ), 0 ] );
    yp( :, :, i, setdiff(1:nt, flipped) ) = circshift( yp_1channels(:,:,1,setdiff(1:nt, flipped)), [ 0, -disperse( i ), 0 ] );

    %yp( :, :, i, : ) = circshift( yp_1channels, [ 0, -disperse( i ), 0, 0 ] );
end

yp = yp.*Cs; % 4D * 4D
f = sum(yp,4); % DUKE: 4D -> 3D
end

