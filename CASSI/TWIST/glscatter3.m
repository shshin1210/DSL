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
% function glscatter3(inp, color)
function glscatter3(inp, color)
% draw 3d points in the OpenGL coordinate system (right-hand rule)
    if nargin<2
        color = 'r';
    end
    
    % note in matlab z is y, y is z.
    % and z should be inverted
    X = inp(:,1); %x
    Y = inp(:,3); %z
    Z = inp(:,2); %y
    
    % example labels
    textCell = arrayfun(@(x,y,z) sprintf('  (%3.2f, %3.2f, %3.2f)',x,y,z),X,Y,Z,'un',0);
    
    %figure;
    scatter3(X,Y,Z,'o', 'MarkerEdgeColor','k', 'MarkerFaceColor',color);
    %surf(X,Y,Z);
    grid on;
    set(gca,'YDir','rev');
    
    xlabel(gca,'X');
    ylabel(gca,'Z');
    zlabel(gca,'Y');

    % Add textCell
    for ii = 1:numel(X) 
        text(X(ii), Y(ii), Z(ii), textCell{ii},'FontSize',8) 
    end
    
    
end
    