clear all;
close all;
warning off;

dat = 'rgb';

for i = 1:3
    
    file_name = fullfile(sprintf('./%s_curve.mat', dat(i)) );
    load(file_name);
end

rfit = createFit(r_x, r_y);
gfit = createFit(g_x, g_y);
bfit = createFit(b_x, b_y);


file_name_out = fullfile(sprintf('./rgb_fit.mat') );


p = [[rfit.a, rfit.b, rfit.c],[gfit.a, gfit.b, gfit.c],[bfit.a, bfit.b, bfit.c]];

save(file_name_out, 'p')