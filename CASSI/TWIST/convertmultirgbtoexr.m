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
% function convertmultirgbtoexr(nickname)
function convertmultirgbtoexr(nickname)
    if nargin<1
        disp enter the nickname of the model
        return
    end

    [cR, cG, cB, rgb] = readconvert3ch([nickname '-F.jpg'],1);

    
    mkdir('hyperspectral_textures');

    dirnames = {'359_364_369nm','375_382_389nm','397_405_415nm','425_437_450nm','464_480_497nm','516_523_530nm','537_544_552nm','560_568_577nm','586_595_604nm','614_624_635nm','645_657_668nm','680_692_705nm','718_732_746nm','761_776_791nm','807_824_841nm','859_878_897nm','916_937_958nm','980_1002_0nm'};

    channelnames = {'w359nm','w364nm','w369nm','w375nm','w382nm','w389nm','w397nm','w405nm','w415nm','w425nm','w437nm','w450nm','w464nm','w480nm','w497nm','w516nm','w523nm','w530nm','w537nm','w544nm','w552nm','w560nm','w568nm','w577nm','w586nm','w595nm','w604nm','w614nm','w624nm','w635nm','w645nm','w657nm','w668nm','w680nm','w692nm','w705nm','w718nm','w732nm','w746nm','w761nm','w776nm','w791nm','w807nm','w824nm','w841nm','w859nm','w878nm','w897nm','w916nm','w937nm','w958nm','w980nm','w1002nm'};

    shellNames = [{'R','G','B'}, channelnames];
    shellDatas{1} = cR;
    shellDatas{2} = cG;
    shellDatas{3} = cB;

    for i=0:size(dirnames,2)-2
        underdir = dirnames{i+1};
        [cR, cG, cB, rgb] = readconvert3ch(['./' underdir '/' nickname '-F.jpg'],1);

        fn1 = [underdir '/' channelnames{3*i+1} '.png'];
        fn2 = [underdir '/' channelnames{3*i+2} '.png'];
        fn3 = [underdir '/' channelnames{3*i+3} '.png'];

        imwrite(rgb(:,:,1),fn1);
        imwrite(rgb(:,:,2),fn2);
        imwrite(rgb(:,:,3),fn3);

        disp([underdir ': ' num2str(size(rgb))]);

        movefile(fn1,['hyperspectral_textures'],'f');
        movefile(fn2,['hyperspectral_textures'],'f');
        movefile(fn3,['hyperspectral_textures'],'f');

        shellDatas{3*(i+1)+1} = cR; disp(['data: ' num2str(size(shellDatas{3*(i+1)+1}))]);
        shellDatas{3*(i+1)+2} = cG; disp(['data: ' num2str(size(shellDatas{3*(i+1)+2}))]);
        shellDatas{3*(i+1)+3} = cB; disp(['data: ' num2str(size(shellDatas{3*(i+1)+3}))]);
    end

    % for the last two channels.
    i = i+1;
    underdir = dirnames{i+1};
    [cR, cG, cB] = readconvert3ch(['./' underdir '/' nickname '-F.jpg'],1);

    fn1 = [underdir '/' channelnames{3*i+1} '.png'];
    fn2 = [underdir '/' channelnames{3*i+2} '.png'];

    imwrite(rgb(:,:,1),fn1);
    imwrite(rgb(:,:,2),fn2);

    disp([underdir ': ' num2str(size(rgb))]);

    movefile(fn1,['hyperspectral_textures'],'f');
    movefile(fn2,['hyperspectral_textures'],'f');

    shellDatas{3*(i+1)+1} = cR; disp(['data: ' num2str(size(shellDatas{3*(i+1)+1}))]);
    shellDatas{3*(i+1)+2} = cG; disp(['data: ' num2str(size(shellDatas{3*(i+1)+2}))]);

    disp(['shellNames: ' num2str(size(shellNames))]);
    disp(['shellDatas: ' num2str(size(shellDatas))]);
    
    cData = containers.Map(shellNames, shellDatas);

    exrwritechannels([nickname '-F.exr'],'zip','half',cData);
end