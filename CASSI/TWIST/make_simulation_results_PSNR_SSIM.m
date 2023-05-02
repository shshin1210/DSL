

%% Init

root_dir = './simulation_results/1_psnr_table';
gt_dir = fullfile( root_dir, 'cassi' );

dataset_name = {'balloons_ms', ...
                'beads_ms', ...
                'cd_ms', ...
                'chart_and_stuffed_toy_ms', ...
                'egyptian_statue_ms', ...
                'face_ms', ...
                'fake_and_real_beers_ms', ...
                'fake_and_real_lemon_slices_ms', ...
                'fake_and_real_strawberries_ms', ...
                'fake_and_real_sushi_ms',...
                'stuffed_toys_ms',...
                'imgd4' };
            
n_data = length( dataset_name );
n_sample_images = [ 1, 2, 3, 5, 9 ];

%% Compute PSNR & SSIM
PSNR_mat = zeros( n_data, length( n_sample_images ), 25 );
SSIM_mat = zeros( n_data, length( n_sample_images ), 25 );

for i = 1:n_data
    name = dataset_name{ i };
    %% Load GT
    gt_dir_data = fullfile( gt_dir, name, 'gt' );
    filename = sprintf( '%s_gt.mat', name );
    gt = load( fullfile( gt_dir_data, filename ) );
    [ height, width, n_channels ] = size( gt.x_twist );
    
    %% Load each type
    j_counter = 1;
    for j = n_sample_images
        our_dir = sprintf( 'recon_%d_samples', j);
        our_dir = fullfile( root_dir, our_dir, name );
        filename = sprintf( '%s_recon_spectrum.mat', name );
        ours = load( fullfile( our_dir, filename ) );
       
       %% for each channel
        for k = 1:n_channels
            gt_1_ch = gt.x_twist( :, :, k );
            ours_1_ch = double( ours.x_twist( :, :, k ) );
            %% Normalize
            max_val = max( gt_1_ch(:) );
            min_val = min( gt_1_ch(:) );
            gt_1_ch = ( gt_1_ch - min_val )/( max_val - min_val );
            
            max_val = max( ours_1_ch(:) );
            min_val = min( ours_1_ch(:) );
            ours_1_ch = ( ours_1_ch - min_val )/( max_val - min_val );
            %% Compute PSNR
            peak_snr = psnr( ours_1_ch, gt_1_ch );
            PSNR_mat( i, j_counter, k ) = peak_snr;
            
            %% Compute SSIM
            structural_sim = ssim( ours_1_ch, gt_1_ch );
            SSIM_mat( i, j_counter, k ) = structural_sim;
        end
        j_counter = j_counter + 1;
    end
end
%% AVG
AVG_PSNR_mat = mean( PSNR_mat, 3 );
AVG_SSIM_mat = mean( SSIM_mat, 3 );

%% Write output
save( fullfile( root_dir, 'PSNR_SSIM.mat' ),...
                                'PSNR_mat', 'SSIM_mat',...
                                'AVG_PSNR_mat', 'AVG_SSIM_mat', 'dataset_name' );