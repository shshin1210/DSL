function [ out ] = cross_shrinkage( p, alpha, params )
%CROSS_SHRINKAGE Cross-shrinkage function (soft-thresholding) to solve L1 term
% Cross-shrinkage function (soft-thresholding) to solve L1 term with constants
%                                     theta*beta
%  prox(p)                = max( 1 - ------------, 0 ) .* p + alpha
%   theta||. - alpha||_1              |p - alpha|
% 
% Reference: Heide et al., The supplemental material of "Encoded diffractive optics for full-spectrum computational imaging", Scientific Reports, 2016
% 
% [ out ] = CROSS_SHRINKAGE( p, alpha, params )
% Inputs
%   p: input matrix in any dimension
%   alpha: constant matrix in the same dimension of p (or scalar)
%   params: parameters
%       params.theta: parameter imposing L1 error
%       params.beta: parameter in thresholding
% Outputs
%   out: shrunken output matrix

out = max(1 - params.theta*params.beta ./ max(abs(p-alpha), 0.001), 0) .* p + alpha;

end

