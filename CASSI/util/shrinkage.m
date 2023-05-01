function [ out ] = shrinkage( p, params )
%SHRINKAGE Shrinkage function (soft-thresholding) to solve L1 term
% Shrinkage function (soft-thresholding) to solve L1 term
%                              theta*beta
%  prox(p)         = max( 1 - ------------, 0 ) .* p
%   theta||.||_1                  |p|
% 
% Reference: Heide et al., The supplemental material of "Encoded diffractive optics for full-spectrum computational imaging", Scientific Reports, 2016
% 
% [ out ] = shrinkage( p, params )
% Inputs
%   p: input matrix in any dimension
%   params: parameters
%       params.theta: parameter imposing L1 error
%       params.beta: parameter in thresholding
% Outputs
%   out: shrunken output matrix

out = max(1 - params.theta * params.beta ./ abs(p), 0) .* p;
% out = max(1 - params.theta * params.beta ./ max(abs(p),0.001), 0) .* p;
% out = max(1 - params.theta * params.beta ./ max(abs(p),1e-3), 0) .* p;
% out = max(1 - params.theta * params.beta ./ max(abs(p),0), 0) .* p;
end

