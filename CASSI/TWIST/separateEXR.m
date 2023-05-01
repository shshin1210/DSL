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
% this code replaces the previous color map in the EXR.
% THIS CODE IS NOT COMPLETE
% function separateEXR(fnexr)
function separateEXR(fnexr)
% read file
cData = exrreadchannels(fnexr);
allKeys = keys(cData);

dirnames = {'359_364_369nm','375_382_389nm','397_405_415nm','425_437_450nm','464_480_497nm','516_523_530nm','537_544_552nm','560_568_577nm','586_595_604nm','614_624_635nm','645_657_668nm','680_692_705nm','718_732_746nm','761_776_791nm','807_824_841nm','859_878_897nm','916_937_958nm','980_1002_0nm'};

channelnames = {'w359nm','w364nm','w369nm','w375nm','w382nm','w389nm','w397nm','w405nm','w415nm','w425nm','w437nm','w450nm','w464nm','w480nm','w497nm','w516nm','w523nm','w530nm','w537nm','w544nm','w552nm','w560nm','w568nm','w577nm','w586nm','w595nm','w604nm','w614nm','w624nm','w635nm','w645nm','w657nm','w668nm','w680nm','w692nm','w705nm','w718nm','w732nm','w746nm','w761nm','w776nm','w791nm','w807nm','w824nm','w841nm','w859nm','w878nm','w897nm','w916nm','w937nm','w958nm','w980nm','w1002nm'};


mkdir('hyperspectral_exr_textures');

% name changed
fnexrout = regexprep(fnexr, '.exr', '_rgb.exr','ignorecase');
fnexrout = ['hyperspectral_exr_textures' '/' fnexrout '.exr'];
% enter shell names
shellNames = {'R','G','B'};
% enter data
shellDatas{1} = cData('R');
shellDatas{2} = cData('G');
shellDatas{3} = cData('B');
% pack the cData
cData_out = containers.Map(shellNames, shellDatas);
% write data
exrwritechannels(fnexrout,'zip','half',cData_out);
delete cData_out;

for i=0:size(dirnames,2)-2
    % name changed
    fnexrout = regexprep(fnexr, '.exr', ['_' dirnames{i+1} '.exr'],'ignorecase');
    fnexrout = ['hyperspectral_exr_textures' '/' fnexrout];
    % enter shell names
    shellNames{1} = allKeys{3*i+1};
    shellNames{2} = allKeys{3*i+2};
    shellNames{3} = allKeys{3*i+3};
    % enter data
    shellDatas{1} = cData(allKeys{3*i+1});
    shellDatas{2} = cData(allKeys{3*i+2});
    shellDatas{3} = cData(allKeys{3*i+3});
    % pack the cData
    cData_out_each = containers.Map(shellNames, shellDatas);
    % write data
    exrwritechannels(fnexrout,'zip','half',cData_out_each);
    delete cData_out_each;
end

% for the last two channels.
i = i+1;

% name changed
fnexrout = regexprep(fnexr, '.exr', ['_' dirnames{i+1} '.exr'],'ignorecase');
fnexrout = ['hyperspectral_exr_textures' '/' fnexrout '.exr'];
% enter shell names
shellNames{1} = allKeys{3*i+1};
shellNames{2} = allKeys{3*i+2};
% enter data
shellDatas{1} = cData(allKeys{3*i+1});
shellDatas{2} = cData(allKeys{3*i+2});
% pack the cData
cData_out_each = containers.Map(shellNames, shellDatas);
% write data
exrwritechannels(fnexrout,'zip','half',cData_out_each);
delete cData_out_each;
    