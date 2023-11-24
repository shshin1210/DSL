x_recon = load('x_recon.mat').x_recon;

n1 = 580;
n2 = 890;
m = 24;
result_path = './Results/';

wvls2b = 430:10:660;

x_recon = reshape(x_recon, [n1, n2, m])*m;
params.lambdas = wvls2b;
params.illuminant = 'd65';
params.cmf = 'Judd_Vos';
srgb_recon = spec2srgb(x_recon, params);

imwrite(srgb_recon*1.1, strcat(result_path, "srgb_recon_ours.png"));
