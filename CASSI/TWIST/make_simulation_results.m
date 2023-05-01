
%% Extract 9 random masks
close all;
clear;
% make_simulation_results_0_random_masks;
% return;
% return;


%% PNSR Table
clear, close all;
input_dir = 'data/';
dataset_name = {'stuffed_toys_ms'};
input_code_dir = './simulation_results/0_extract_9_random_masks';
input_dispersion_csv = './simulation_results/synthetic_disperse.csv';
output_dir_for_synthetic_cassi = './simulation_results/1_psnr_table/cassi';
output_dir = './simulation_results/1_psnr_table';
flipped = [2];
make_simulation_results_1_PNSR_table;


% %% Color Patch
% clear, close all;
% 
% %% Spectrum Graph
% clear, close all;
% 
% %% CV & RMSE
% clear, close all;