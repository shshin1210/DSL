%============================================================================
%
% Codename: Bird Scanner (Yale Computer Graphics Group)
%
% Copyright (C) 2011-12 Min H. Kim. All rights reserved.
%
% GNU General Public License Usage
% Alternatively, this file may be used under the terms of the GNU General
% Public License version 3.0 as published by the Free Software Foundation
% and appearing in the file LICENSE.GPL included in the packaging of this
% file. Please review the following information to ensure the GNU General
% Public License version 3.0 requirements will be met:
% http://www.gnu.org/copyleft/gpl.html.
%
%============================================================================

% principal component analysis (one chromaticity)

uv = imread('1_global_uv_20sec_f11_f0.4m.tif');
rd = imread('1_global_rgb_0.1sec_f11_f0.4m-R.tif');
gr = imread('1_global_rgb_0.1sec_f11_f0.4m-G.tif');
bl = imread('1_global_rgb_0.1sec_f11_f0.4m-B.tif');
ir = imread('1_global_ir_0.1sec_f11_f0.4m.tif');


% boolean mask
MASK = zeros(size(gr),'uint8');
MASK(1901:2500,1101:2000) = 1; % 600-by-900
imshow(MASK.*255);

% gamma = 1.8; % original encoded
% 
% luv = uint16(65535.*((double(uv)./65535).^gamma));
% lrd = uint16(65535.*((double(rd)./65535).^gamma));
% lgr = uint16(65535.*((double(gr)./65535).^gamma));
% lbl = uint16(65535.*((double(bl)./65535).^gamma));
% lir = uint16(65535.*((double(ir)./65535).^gamma));


S = cat(3, uv,bl,gr,rd,ir);
%S = cat(3, luv,lbl,lgr,lrd,lir);

[X, R] = imstack2vectors(S, MASK);
h = 600;
w = 900;
% [X, R] = imstack2vectors(S);
% w = size(gr,2);
% h = size(gr,1);

f1 = reshape(X(:,1), h,w);figure;imshow(f1);set(gcf,'name','org-1');
f2 = reshape(X(:,2), h,w);figure;imshow(f2);set(gcf,'name','org-2');
f3 = reshape(X(:,3), h,w);figure;imshow(f3);set(gcf,'name','org-3');
f4 = reshape(X(:,4), h,w);figure;imshow(f4);set(gcf,'name','org-4');
f5 = reshape(X(:,5), h,w);figure;imshow(f5);set(gcf,'name','org-5');

% subtract 4 or 5
base = f4; % 4 is best
DE1 = abs(double(f3) - double(base));  De1 = 1-(DE1/max(max(DE1)));figure;imshow(De1);set(gcf,'name','chroma3-4');
DE2 = abs(double(f2) - double(base));  De2 = 1-(DE2/max(max(DE1)));figure;imshow(De2);set(gcf,'name','chroma2-4');
DE3 = abs(double(f1) - double(base));  De3 = 1-(DE3/max(max(DE1)));figure;imshow(De3);set(gcf,'name','chroma1-4');
DE4 = abs(double(f5) - double(base));  De4 = 1-(DE4/max(max(DE1)));figure;imshow(De4);set(gcf,'name','chroma5-4');

gamma = 1/2.2;

dh1 = histeq(De1).^gamma; figure; imshow(dh1);set(gcf,'name','d-histeq-1');
dh2 = histeq(De2).^gamma; figure; imshow(dh2);set(gcf,'name','d-histeq-2');
dh3 = histeq(De3).^gamma; figure; imshow(dh3);set(gcf,'name','d-histeq-3');
dh4 = histeq(De4).^gamma; figure; imshow(dh4);set(gcf,'name','d-histeq-4');


%S2 = cat(3,dh1, dh2, dh3, dh4);
S2 = cat(3,dh1, dh2, dh3);
[X, R] = imstack2vectors(S2);
P = princomp(X,3);
d = diag(P.Cy);
figure; plot(d);

P = princomp(X,2); % 2 is best
h1 = P.X(:,1);h1 = reshape(h1, h,w);figure;imshow(h1./max(max(h1)));set(gcf,'name','pca-1-1');
h2 = P.X(:,2);h2 = reshape(h2, h,w);figure;imshow(h2./max(max(h2)));set(gcf,'name','pca-1-2');
h3 = P.X(:,3);h3 = reshape(h3, h,w);figure;imshow(h3./max(max(h3)));set(gcf,'name','pca-1-3'); % dont use
% h4 = P.X(:,4);h4 = reshape(h4, h,w);figure;imshow(h4./max(max(h4)));set(gcf,'name','pca-1-4');

% averaging
hall = (h1+h2)./2;
%final = (h1./max(max(h1))).^(1/2.2);
final = hall./max(max(hall));
final = histeq(final).^(1/2.2);
%imwrite(final,'result1.png');
figure;imshow(final);set(gcf,'name','final');

submask = histeq(double(f4)).^(1/2.2);
final2 = 1-abs(final - submask);
imwrite(final2,'result.png');
figure;imshow(final2);set(gcf,'name','final2');

% this doesn't help!
%
% H = fspecial('unsharp',1.0);
% final3 = imfilter(final2,H,'replicate');
% imwrite(final3,'result3.png');
% figure;imshow(final3);set(gcf,'name','final3');
