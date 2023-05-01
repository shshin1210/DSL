function [ out ] = div( in )
%DIV Divergence function
% Divergence function for total variational problems.
% 
% gradient.transpose(V) = -div(V)
%              dV1   dV2
% divergence = --- + ---
%              dx    dy
% 
% [ out ] = div( in )
% Inputs
%   in: input gradient fields of an image (h,w,d*2)
%       Each pair planes(1,2;3,4;5,6) describes gradient fields of each channel of an RGB image.
%       e.g., in(:,:,1:2) == cat(3,dx(img(:,:,1)),dy(img(:,:,1)))
% Outputs
%   out: output divergence map (h,w,d)

[h,w,d] = size(in);
out = zeros(h,w,d/2);

for i = 1:(d/2)
    in_onechannel = in(:,:,(2*i-1):(2*i));
    inx = [in_onechannel(:,1,1), in_onechannel(:,2:w-1,1) - in_onechannel(:,1:w-2,1), -in_onechannel(:,w-1,1)];
    iny = [in_onechannel(1,:,2); in_onechannel(2:h-1,:,2) - in_onechannel(1:h-2,:,2); -in_onechannel(h-1,:,2)];
    
    out(:,:,i) = inx + iny;
end

end

