path = 'CZ_hsdbi/';
files = dir([path '/*.mat']);

for file = files'
    load([path file.name]);
    
    
    wvls2b = 420:10:720;

    sRGB = R2sRGB(wvls2b,ref,0, 0, 0);
    
    [pathstr,name,ext] = fileparts(file.name);
    
    imwrite(sRGB, [path name '.png']);
end