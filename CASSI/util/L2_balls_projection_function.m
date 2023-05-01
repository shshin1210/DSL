function [ out ] = L2_balls_projection_function( in )
%L2_BALLS_PROJECTION_FUNCTION Pointwise Euclidean projectors onto L2 balls
% Pointwise Euclidean projectors onto L2 balls,
% which is a solution of proximal gradient descent of an indicator function.
% Reference: Chambolle et al., "A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging", J Math Imaging Vis (2011)
% 
%            in
% out = -------------
%        max(1,|in|)
% 
% [ out ] = L2_balls_projection_function( in )
% Inputs
%   in: input vector (len,1)
% Outputs
%   out: output vector projected onto L2 balls

out = in ./ max(1, abs(in));
end

