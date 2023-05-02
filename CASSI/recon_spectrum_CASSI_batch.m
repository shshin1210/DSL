algo_fun = @(data_path, result_path) recon_spectrum_CASSI_single(data_path, result_path);

data_dir = './data';
result_dir = './results';

files=dir(data_dir);
files = files(3:end); % remove . and .. folders

for i = 1:numel(files)
    file = files(i);
    data_path = [data_dir '/' file.name '/'];
    result_path = [result_dir '/'  file.name '/'];

    if ~exist(result_path, 'dir')
        mkdir(result_path);
    end
    
    
    fprintf('Current data: %s\n', data_path);
    
    cp = tic();
    algo_fun(data_path, result_path);
    toc(cp);

end
