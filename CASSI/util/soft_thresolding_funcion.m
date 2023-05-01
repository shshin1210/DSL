function [ out ] = soft_thresolding_funcion( in, bottom, threshold )
%SOFT_THRESOLDING_FUNCION Soft-thresholding function for minimizing L1 error term
% Soft-thresholding function for minimizing L1 error term
% 
%                     /
%                    /
%                   /
%  -threshold      /
%     -------------    bottom
%    /       +threshold
%   /
%  /
% /
% 
% The bottom value is used for primal-dual algorithms. (Chambolle et al., "A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging", J Math Imaging Vis (2011))
% 
% [ out ] = soft_thresolding_funcion( in, bottom, threshold )
% Inputs
%   in: input map
%   bottom: bottom value map as shown above
%   threshold: threshold value map for soft-thresholding
% Outputs
%   out: soft-thresholded output value map

dif = in - bottom;
out = ...
      (in - threshold) .* (dif > threshold)...
    + (in + threshold) .* (dif < -threshold)...
    + bottom .* (abs(dif) <= threshold);

end

