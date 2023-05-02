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
% function convertsrgbtoexr(fnjpg, fnexr)
function convertsrgbtoexr(fnjpg, fnexr)
% this code replaces the previous color map in the EXR.

fnexrout = regexprep(fnjpg, '.jpg', '_out.exr','ignorecase');

[cR, cG, cB, rgb] = readconvert3ch(fnjpg,1);


if nargin<2
    shellNames = {'R','G','B'};
    shellDatas{1} = cR;
    shellDatas{2} = cG;
    shellDatas{3} = cB;
    cData = containers.Map(shellNames, shellDatas);
else
    cData = exrreadchannels(fnexr);
    cData('R') = cR;
    cData('G') = cG;
    cData('B') = cB;
end

exrwritechannels(fnexrout,'zip','half',cData);

end