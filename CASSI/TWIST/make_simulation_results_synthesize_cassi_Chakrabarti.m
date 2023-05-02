close all;
clear;

%% init
input_dir = '/media/vclab-office-01/Study/Inputs/2D_Hyperspectral_Image_Data_Chakrabarti/CZ_hsdb';
dataset_name = {'imgd4'};
input_code_dir = './simulation_results/0_extract_9_random_masks';
input_dispersion_csv = './simulation_results/synthetic_disperse.csv';
output_dir_for_synthetic_cassi = './simulation_results/1_psnr_table/cassi';
output_dir = './simulation_results/1_psnr_table';

filename_ordering = { '2_2', ...
                      '2_1', ...
                      '2_3', ...
                      '1_2', ...
                      '3_2', ...
                      '1_1', ...
                      '1_3', ...
                      '3_1', ...
                      '3_3' };

%% Load Synthetic Coded Masks

n_coded_masks = length( filename_ordering );
coded_masks = cell( n_coded_masks, 1 );

for i = 1:n_coded_masks
    filename = sprintf( 'cell_%s.png', filename_ordering{i} );
    filename = fullfile( input_code_dir, filename );
    coded_masks{i} = imread( filename );
end

[ height, width ] = size( coded_masks{ i } );

%% Load Dispersion
disperse_csv = csvread( input_dispersion_csv );
wvls = disperse_csv(:, 1);
disperse = disperse_csv(:, 2);
fittedmodel = fit(wvls, disperse, 'power2');
ref_disperse = fittedmodel(636);
% wvls2b = [450:10:690];
%% Do synthesize
n_data = length( dataset_name );
spectrum_start_idx = 4;                 % 450 nm
spectrum_end_idx = 28;                  % 690 nm
n_spectums = spectrum_end_idx - spectrum_start_idx + 1;
wvls...
    = ( 420 + ( spectrum_start_idx - 1 )*10 ):10:( 420 + ( spectrum_end_idx - 1 )*10 );
disperse = int8( fittedmodel( wvls ) - ref_disperse );
for i = 1:n_data
    %% Load all spectrums
    spectrums = zeros( height, width, n_spectums );
    name = dataset_name{ i };
    filename = sprintf( '%s/%s.mat', input_dir, name );
    load( filename );
    for j = 1:n_spectums
       idx = spectrum_start_idx + ( j - 1 );
       spectra = ref( :, :, idx );
      %% resize
%        cond = ( width > height );
%        scaled_size = cond*width  + ( ~cond )*height;
%        offset = int32( abs( height - width )/2 );
%        spectra = imresize( spectra, [ scaled_size, scaled_size ], 'bicubic' );
%        if strcmp( name, 'stuffed_toys_ms' ) == 1
%            offset = scaled_size - height + 1;
%            spectra = spectra( offset:end, : );
%        else
%            if cond
%                spectra = spectra( offset:( end - offset - 1 ), : );
%            else
%                spectra = spectra( :, offset:( end - offset - 1 ) );
%            end
%            
%        end
       spectra = imresize( spectra, [ height, width ]);
       spectrums( :, :, j ) = spectra;
    end
%     spectrums = spectrums/(2^16 - 1);
    %% Save Spectrum
    out_dir = fullfile( output_dir_for_synthetic_cassi, name, 'gt' );
     if ~exist( out_dir, 'dir' )
           mkdir( out_dir );
       end
    filename = sprintf( '%s/%s_gt.mat', out_dir, name );
    x_twist = spectrums;
    wvls2b = wvls;
    save( filename, 'x_twist', 'wvls2b' );
    [ refwhite, sRGB ] = cassidisplay_for_simulation( 0, 0, filename );
    filename = sprintf( '%s/%s_sRGB_gt.png', out_dir, name );
    imwrite( sRGB, filename );
    
    for j= 1:n_spectums
        filename = sprintf( '%s/%dnm_gt.png', out_dir, wvls( j ) );
        imwrite( spectrums( :, :, j )*255, filename );
    end
    
    
    %% For each mask
    for j = 1:n_coded_masks
       %% Load mask
        mask = single( coded_masks{j} )/255;
        mask = repmat( mask, 1, 1, n_spectums );
        %% Punch spectrums
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
%         figure, imshow( img );
        
       %% Save
       out_dir = fullfile( output_dir_for_synthetic_cassi, name );
       if ~exist( out_dir, 'dir' )
           mkdir( out_dir );
       end
       filename = sprintf( '%s/%s_%02d.png', out_dir, name, j );
       imwrite( img, filename );
    end
end