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
% function exr2bird(fn, refwhite)
function exr2bird(fn, refwhite)
% exr to bird cones
    
    cData  = exrreadchannels(fn);
    
    wvls = [359.4800,364.3709,369.7441,375.6510,382.1483,389.2990,397.1725,405.8457,415.4038,425.9410,437.5615,450.3805,464.5258,480.1385,497.3748,516.4075,523.1835,530.1877,537.4280,544.9125,552.6496,560.6479,568.9164,577.4644,586.3013,595.4373,604.8824,614.6475,624.7432,635.1812,645.9729,657.1307,668.6671,680.5950,692.9279,705.6796,718.8647,732.4978,746.5944,761.1704,776.2422,791.8268,807.9419,824.6056,841.8367,859.6547,878.0798,897.1328,916.8351,937.2090,958.2778,980.0651,1002.596;];
    channelnames = {'w359nm','w364nm','w369nm','w375nm','w382nm','w389nm','w397nm','w405nm','w415nm','w425nm','w437nm','w450nm','w464nm','w480nm','w497nm','w516nm','w523nm','w530nm','w537nm','w544nm','w552nm','w560nm','w568nm','w577nm','w586nm','w595nm','w604nm','w614nm','w624nm','w635nm','w645nm','w657nm','w668nm','w680nm','w692nm','w705nm','w718nm','w732nm','w746nm','w761nm','w776nm','w791nm','w807nm','w824nm','w841nm','w859nm','w878nm','w897nm','w916nm','w937nm','w958nm','w980nm','w1002nm'};

    for i=1:size(channelnames,2)
        temp = channelnames(i);
        R(:,:,i) = cData(temp{1});
    end
    
    
    Rw = R2WBR(R,refwhite);
    
    R_USML = R2BIRD(wvls, Rw);
    Rmax = max(max(max(R_USML)));
    R_USMLn = R_USML./Rmax;
    
    shellNames = {'U','S','M','L'};
    shellDatas{1} = R_USMLn(:,:,1);
    shellDatas{2} = R_USMLn(:,:,2);
    shellDatas{3} = R_USMLn(:,:,3);
    shellDatas{4} = R_USMLn(:,:,4);
    cData = containers.Map(shellNames, shellDatas);
    fn2 = regexprep(fn, '.exr', '_USML.exr','ignorecase');
    exrwritechannels(fn2,'zip','half',cData);
    exrinfo(fn2);
    
    shellNamesU = {'R','G','B'};
    shellDatasU{1} = zeros(size(R_USMLn,1),size(R_USMLn,2));
    shellDatasU{2} = zeros(size(R_USMLn,1),size(R_USMLn,2));
    shellDatasU{3} = R_USMLn(:,:,1);
    cData = containers.Map(shellNamesU, shellDatasU);
    fn2 = regexprep(fn, '.exr', '_u.exr','ignorecase');
    exrwritechannels(fn2,'zip','half',cData);
    exrinfo(fn2);
    
    shellNamesL = {'B','G','R'};
    shellDatasL{1} = R_USMLn(:,:,2);
    shellDatasL{2} = R_USMLn(:,:,3);
    shellDatasL{3} = R_USMLn(:,:,4);
    cData = containers.Map(shellNamesL, shellDatasL);
    fn2 = regexprep(fn, '.exr', '_lms.exr','ignorecase');
    exrwritechannels(fn2,'zip','half',cData);
    exrinfo(fn2);
%---------------------------------------------------------------    
    Rmax = max(max(R_USML));
    Rmax = Rmax(:);
    R_USMLn(:,:,1) = R_USML(:,:,1)./Rmax(1);
    R_USMLn(:,:,2) = R_USML(:,:,2)./Rmax(2);
    R_USMLn(:,:,3) = R_USML(:,:,3)./Rmax(3);
    R_USMLn(:,:,4) = R_USML(:,:,4)./Rmax(4);
    
    shellNames = {'U','S','M','L'};
    shellDatas{1} = R_USMLn(:,:,1);
    shellDatas{2} = R_USMLn(:,:,2);
    shellDatas{3} = R_USMLn(:,:,3);
    shellDatas{4} = R_USMLn(:,:,4);
    cData = containers.Map(shellNames, shellDatas);
    fn2 = regexprep(fn, '.exr', '_USML-n.exr','ignorecase');
    exrwritechannels(fn2,'zip','half',cData);
    exrinfo(fn2);
    
    shellNamesU = {'R','G','B'};
    shellDatasU{1} = zeros(size(R_USMLn,1),size(R_USMLn,2));
    shellDatasU{2} = zeros(size(R_USMLn,1),size(R_USMLn,2));
    shellDatasU{3} = R_USMLn(:,:,1);
    cData = containers.Map(shellNamesU, shellDatasU);
    fn2 = regexprep(fn, '.exr', '_u-n.exr','ignorecase');
    exrwritechannels(fn2,'zip','half',cData);
    exrinfo(fn2);
    
    shellNamesL = {'B','G','R'};
    shellDatasL{1} = R_USMLn(:,:,2);
    shellDatasL{2} = R_USMLn(:,:,3);
    shellDatasL{3} = R_USMLn(:,:,4);
    cData = containers.Map(shellNamesL, shellDatasL);
    fn2 = regexprep(fn, '.exr', '_lms-n.exr','ignorecase');
    exrwritechannels(fn2,'zip','half',cData);
    exrinfo(fn2);
end