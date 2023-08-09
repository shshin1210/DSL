clear all;
close all;
warning off;

dat_path = 'C:/Users/owner/Documents/GitHub/Scalable-Hyp-3D-Imaging/dataset/image_formation/dat/method2';
pitch =  7.9559e-06;

file_list = dir(fullfile(dat_path, 'dispersion_coordinates_*.mat' ));

for i = 1:numel(file_list)
    file_name = fullfile(dat_path, file_list(i).name);
    load(file_name);
    x = x / pitch;
    y = y / pitch;
%     xo = xo / pitch;
%     yo = yo / pitch;

    px = dispertion_fit_method2(x, y, xo);
    py = dispertion_fit_method2(x, y, yo);

    xo_recon = px(x,y);
    yo_recon = py(x,y);
    
    figure(1); 
    subplot(1,2,1); plot(xo_recon, yo_recon, '.'); title('recon');
    subplot(1,2,2); plot(xo, yo, '.'); title('GT');
    subtitle(i)
    pause(1);

    error = sqrt((xo_recon-xo).^2 + (yo_recon-yo).^2);

%     figure(1); imagesc(reshape(error, [64, 36])); colorbar; title(sprintf('error %s', file_list(i).name(1:end-4)));
    pause(1);

    fprintf('reprojection error in px\n');
    fprintf('\t mean error:%f, max error:%f, stddev: %f\n', mean(error(:), 'omitnan'), max(error(:)), std(error(:), 'omitnan'));

    file_name_out = fullfile(dat_path, sprintf('param_%s.mat', file_list(i).name(1:end-4)));

    p = [[px.p00, px.p10, px.p01, px.p20, px.p11, px.p02, px.p30, px.p21, px.p12, px.p03, px.p40, px.p31, px.p22, px.p13, px.p04, px.p50, px.p41, px.p32, px.p23, px.p14, px.p05 ],
        [py.p00, py.p10, py.p01, py.p20, py.p11, py.p02, py.p30, py.p21, py.p12, py.p03, py.p40, py.p31, py.p22, py.p13, py.p04, py.p50, py.p41, py.p32, py.p23, py.p14, py.p05 ]
        ];

    save(file_name_out, 'p')
end