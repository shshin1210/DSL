
% BFILTER2 Two dimensional bilateral filtering.
%    This function implements 2-D bilateral filtering using
%    the method outlined in:
%
%       C. Tomasi and R. Manduchi. Bilateral Filtering for
%       Gray and Color Images. In Proceedings of the IEEE
%       International Conference on Computer Vision, 1998.
%
%    B = bfilter2(A,W,SIGMA) performs 2-D bilateral filtering
%    for the grayscale or color image A. A should be a double
%    precision matrix of size NxMx1 or NxMx3 (i.e., grayscale
%    or color images, respectively) with normalized values in
%    the closed interval [0,1]. The half-size of the Gaussian
%    bilateral filter window is defined by W. The standard
%    deviations of the bilateral filter are given by SIGMA,
%    where the spatial-domain standard deviation is given by
%    SIGMA(1) and the intensity-domain standard deviation is
%    given by SIGMA(2).
%
% Douglas R. Lanman, Brown University, September 2006.
% dlanman@brown.edu, http://mesh.brown.edu/dlanman




%          =============== EXAMPLE ===============
% B = bfltGray_using_color_guide(img2,img2,w,sigma(1),sigma(2), mask);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pre-process input and select appropriate filter.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implements bilateral filtering for grayscale images.
function B = bfltter_with_mask(A,cG,w,sigma_d,sigma_r, mask)

% Pre-compute Gaussian distance weights.
[X,Y] = meshgrid(-w:w,-w:w);
G = exp(-(X.^2+Y.^2)/(2*sigma_d^2));

% Create waitbar.
h = waitbar(0,'Applying bilateral filter...');
set(h,'Name','Bilateral Filter Progress');

% Apply bilateral filter.

inds = find(mask==1);
dim = size(A);
B = zeros(dim);

for i = 1:numel(inds)
    ind = inds(i);
    [r,c] = ind2sub([size(A,1), size(A,2)], ind);
    
    % Extract local region.
    iMin = max(r-w,1);
    iMax = min(r+w,dim(1));
    jMin = max(c-w,1);
    jMax = min(c+w,dim(2));
    I = A(iMin:iMax,jMin:jMax,:);
    cI = cG(iMin:iMax,jMin:jMax,:);
    M = mask(iMin:iMax,jMin:jMax);
    
    dr = cI(:,:,1)-cG(r,c,1);
    dg = cI(:,:,2)-cG(r,c,2);
    db = cI(:,:,3)-cG(r,c,3);
    
    % Compute Gaussian intensity weights.
    H = M.* exp(-(dr.^2+dg.^2+db.^2)/(2*sigma_r^2));
    
    % Calculate bilateral filter response.
    F = H.*G((iMin:iMax)-r+w+1,(jMin:jMax)-c+w+1);
    
    for j = 1:size(A,3)
        B(r,c,j) = sum(sum(F(:).* reshape(I(:,:,j),[],1))) / sum(F(:));
    end
    waitbar(i/numel(inds));
end

% Close waitbar.
close(h);


