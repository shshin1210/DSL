clear;
close all;
%% load image
% filename = ['/media/vclab-office-01/Study/Inputs/2d_Hyperspectral_Image_Data_Columbia/fake_and_real_lemon_slices_ms'...
%             '/fake_and_real_lemon_slices_ms/fake_and_real_lemon_slices_RGB.bmp'];
%         
% img = double( imread( filename ) )/255;
% figure, imshow( img ), title( 'before' );
% 
% % p1: 108, 61 ->   110, 50
% % p2: 377,54  ->   370, 50
% % p3: 386,224 ->   370, 210
% % p4: 110,233 ->   110, 210

%% Make transformation
T = maketform( 'projective',[ 108 61; 377 54; 386 224; 110 233 ],...
                            [ 110 50; 370 50; 370 210; 110 210 ]);
                        
R = makeresampler('cubic','fill');
% K = imtransform( img, T, R, 'XYScale', 1 );
% figure, imshow(K), title( 'after' );

%% Warp .mat file
% filename = './simulation_results/1_psnr_table/cassi/fake_and_real_lemon_slices_ms/gt/fake_and_real_lemon_slices_ms_gt.mat';
% out_filname = './simulation_results/1_psnr_table/cassi/fake_and_real_lemon_slices_ms/gt/fake_and_real_lemon_slices_warped_ms_gt.mat';
filename = './simulation_results/1_psnr_table/recon_9_samples/fake_and_real_lemon_slices_ms/fake_and_real_lemon_slices_ms_recon_spectrum.mat';
out_filname = './simulation_results/1_psnr_table/recon_9_samples/fake_and_real_lemon_slices_ms/fake_and_real_lemon_slices_ms_recon_spectrum_warped.mat';

load( filename );
[ height, width, n_channels ] = size( x_twist );

for i = 1:n_channels
    img = x_twist( :, :, i );
    K = imtransform( img, T, R, 'XYScale', 1, 'SIZE', size(img) );
%     figure, imshow( K );
    x_twist2( :, :, i ) = K;
end
x_twist = x_twist2;
save( out_filname, 'wvls2b', 'x_twist' );

[refwhite,sRGB] = cassidisplay_for_simulation( 0, 0, out_filname );
% out_filname = './simulation_results/1_psnr_table/cassi/fake_and_real_lemon_slices_ms/gt/fake_and_real_lemon_slices_warped_ms_gt.png';
out_filname = './simulation_results/1_psnr_table/recon_9_samples/fake_and_real_lemon_slices_ms/fake_and_real_lemon_slices_ms_recon_spectrum_warped.png';
imwrite( sRGB, out_filname );

%% Warp an Image


%% Warp Images