close all;
clear

%% Init
input_dir = '/media/vclab-office-01/Study/Inputs/2d_Hyperspectral_Image_Data_Columbia';
scene_name = 'chart_and_stuffed_toy_ms';
input_dispersion_csv = './simulation_results/synthetic_disperse.csv';
out_dir = './simulation_results/5_comp_sCassi';
new_dir = fullfile( out_dir, scene_name );
if ~exist( new_dir, 'dir' )
    mkdir( new_dir );
end

mCassi_height = 501;
nCassi_width = 645;
scale_factor = 3;
sCassi_height = mCassi_height*scale_factor;
sCassi_width = nCassi_width*scale_factor;

%% Make a coded Mask
% img_coded_mask = imread( './simulation_results/coded.png' );
img_coded_mask = imread( './simulation_results/new_coded.png' );
pixel_ratio = 0.5;
intermediate_height = int32( sCassi_height*pixel_ratio );
intermediate_width = int32( sCassi_width*pixel_ratio );
padding = 120;

rectangle...
        = int32([ padding padding intermediate_width intermediate_height ]);
img_coded_mask = imcrop( img_coded_mask, rectangle );
% figure, imshow( img_coded_mask );
img_coded_mask...
            = imresize( img_coded_mask, [ sCassi_height, sCassi_width ], 'nearest' );
% figure, imshow( img_coded_mask );
imwrite( img_coded_mask, sprintf('%s/%s.png', out_dir, 'sCassi_coded_mask' ) );

%% Load Dispersion
disperse_csv = csvread( input_dispersion_csv );
wvls = disperse_csv(:, 1);
disperse = disperse_csv(:, 2);
fittedmodel = fit(wvls, disperse, 'power2');
ref_disperse = fittedmodel(636);

%% Do Synthesize
spectrum_start_idx = 6;                 % 450 nm
spectrum_end_idx = 30;                  % 690 nm
n_spectums = spectrum_end_idx - spectrum_start_idx + 1;
wvls...
    = ( 400 + ( spectrum_start_idx - 1 )*10 ):10:( 400 + ( spectrum_end_idx - 1 )*10 );
disperse = int8( fittedmodel( wvls ) - ref_disperse );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% scale disperse!
disperse = disperse*3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

spectrums = zeros( sCassi_height, sCassi_width, n_spectums );
for j = 1:n_spectums
    %% Load a spectra
     idx = spectrum_start_idx + ( j - 1 );
        filename = sprintf('%s/%s/%s/%s_%02d.png',...
                            input_dir,...
                            scene_name,...
                            scene_name,...
                            scene_name,...
                            idx );
       spectra = imread( filename );
    
    %% Resize
    cond = ( sCassi_width > sCassi_height );
    scaled_size = cond*sCassi_width  + ( ~cond )*sCassi_height;
    offset = int32( abs( sCassi_height - sCassi_width )/2 );
    spectra = imresize( spectra, [ scaled_size, scaled_size ], 'bicubic' );
    
    if cond
        spectra = spectra( offset:( end - offset - 1 ), : );
    else
        spectra = spectra( :, offset:( end - offset - 1 ) );
    end
    
    spectrums( :, :, j ) = spectra;
%     figure, imshow( spectra );
end
spectrums = spectrums/(2^16 - 1);
%% Save Spectrum

filename = sprintf( '%s/sCassi_%s_gt.mat', new_dir, scene_name );
x_twist = spectrums;
wvls2b = wvls;
save( filename, 'x_twist', 'wvls2b' );
[ refwhite, sRGB ] = cassidisplay_for_simulation( 0, 0, filename );
filename = sprintf( '%s/sCassi_%s_sRGB_gt.png', new_dir, scene_name );
imwrite( sRGB, filename );

for j= 1:n_spectums
    filename = sprintf( '%s/sCassi_%s_%dnm_gt.png', new_dir, scene_name, wvls( j ) );
    imwrite( spectrums( :, :, j ), filename );
end

%% Punch Spectrum
mask = single( img_coded_mask )/255;
mask = repmat( mask, 1, 1, n_spectums );
maksed_spectrums = single( spectrums );
maksed_spectrums = maksed_spectrums.*mask;

%% Disperse
for k = 1:n_spectums
    %             figure, imshow( spectrums( :, :, k ) );
    maksed_spectrums( :, :, k ) = circshift( maksed_spectrums( :, :, k ), [ 0, disperse( k ), 0 ] );
end

%% Project
img = sum( maksed_spectrums, 3 );
max_val = max( img(:) );
min_val = min( img(:) );
img = ( img - min_val )/( max_val - min_val );

%% Save
filename = sprintf( '%s/sCassi_%s_cassi.png', new_dir, scene_name );
imwrite( img, filename );



