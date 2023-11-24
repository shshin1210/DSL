algo_fun = @(data_path, data_fn, result_path, result_fn) recon_spectrum_CASSI_single(data_path, data_fn, result_path, result_fn);

data_dir = './Valid_spectral/';
result_dir = './Results/';
file = dir([data_dir '*.mat']);

for i = 1:size(file,1)
    data_fn = file(i).name;
    result_fn = strcat(data_fn(1:end-4),  "_result.mat");
    if isfile(fullfile(result_dir, result_fn)) 
        continue
    end
    
    fprintf('Current data: %s\n', data_fn);
    
    cp = tic();
    algo_fun(data_dir, data_fn, result_dir, result_fn);
    toc(cp);
end
