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
% Acknowlegments
% Portions of this file were based on the original code of David Kittle 
% (Duke University).
%============================================================================
% function cassireconsingle(type, starti)
function cassireconsingle(type, starti)
    %=====================================================================
    % actual height for each segmentation
    % this should be 4, 8, 16, or 32;
    % piecedepth = 8; % for 32bit system
    %piecedepth = 256*2; %64*4; % piece depth % this is too much 
    %piecedepth = 256*8; % this is a reasonable level for 16GB system
    piecedepth = 2048; % from camera pixel height
    %=====================================================================
    % Twist Paramter sets
    %
    if nargin==0
        disp 'enter the type: [vis] or [uv]'
        return;
    end
    % enter the wavelength
    laser_wl = 560;  %DON'T CHANGE THIS
    % fixed by our filter set (given by Duke)

    %------------------------------------------------------------------
    % single iterative denoising calculatin (parameters)    
    % DUKE settings:
    taub=0.1; % for higher resolution image --> it shoul be smaller.
    % piterb=2; % 
    % piterb=8; % took double time than piterb=4, but no improvement in quality
    piterb=4; % this was the best with cassidenoise.m
    %------------------------------------------------------------------
%     taub=12; % too soft
%     piterb=40; % too soft
    %------------------------------------------------------------------
       
    
%    translations=1:60;  

    % Maximum TWIST iteration (from Ver. 5)
    % from Ver.5 it doesn't matter.
    %------------------------------------------------------------------
    % for original CASSI
%    iterationsb=40; % after 40 almost same...        
    %------------------------------------------------------------------
    iterationsb=100; % for single calculation
    %------------------------------------------------------------------
    % gradiant tolerance to stop iterations.
    %tolA = 1e-4; % for version 6
    tolA = 1e-8; % for single snap version
    
    if strcmp(type,'vis')==1 
%        deltab=3; % [BEST] 3 is the least value (before crashing TWIST)
        deltab=2; % [Now it works beautifully for single snap shot]
        start_wvl=420; %DON'T CHANGE THIS
        end_wvl=680; %DON'T CHANGE THIS
        %    start_wvl=400;
        %    end_wvl=700;
    elseif strcmp(type,'uv')==1 
        deltab=5; % [BEST] 5 makes 10bins(10nm) w/o TWIST crashing.
        start_wvl=300; %DON'T CHANGE THIS
        end_wvl=380; %DON'T CHANGE THIS
    end        
    %=====================================================================
    % starting iteration
    if nargin < 2 % for visible spectrum
        starti = 1;
    end

    %=====================================================================
    % Displaying parameters:
    disp(['========================================']);
    disp(['[VARIABLES]']);
    disp(['Spectral Type: ' type]);
    disp(['Piece Depth: ' num2str(piecedepth) 'px']);
    disp(['taub: ' num2str(taub)]);
    disp(['piterb: ' num2str(piterb)]);
    disp(['iterationsb: ' num2str(iterationsb)]);
    disp(['tolA: ' num2str(tolA)]);
    disp(['deltab: ' num2str(deltab)]);
    disp(['start wvl: ' num2str(start_wvl)]);
    disp(['end wvl: ' num2str(end_wvl)]);
    disp(['starting index: ' num2str(starti)]);
    disp(['========================================']);
    disp(['[START]']);
    
    %=====================================================================
    % ask filename for the calibration data
    % file read
    [file_name,file_path] = uigetfile({'*.bmp','bmp';},'Choose a calibration bmp file');
    % leave function when no file is selected.
    if size(file_path,2) == 1
        return;
    end
    sourcecalib = char(strcat(file_path,file_name));   
    
    cdata = imread(sourcecalib);
    height = size(cdata,1);
    width = size(cdata,2);
        
    disp 'Calibration data reading done!';
    %=====================================================================
    % ask filename for the main listfile
    % file read
    [file_name,file_path] = uigetfile({'*.mat','mat';},'Choose a main capture mat file');
    % leave function when no file is selected.
    if size(file_path,2) == 1
        return;
    end
    sourcepath = char(strcat(file_path,file_name));  

    % load the source MAT file
    load(sourcepath);
  
   
    disp 'Main data reading done!';    
    
    disp 'taking only the first single shot'
    shift = shift(:,1);
    I_trans = I_trans(:,:,1);
    
    %=====================================================================
    % first derive source matrices (Cu and I_trans).

    % loading calibration data (fixed by hardware configuration)
    %[mono_input calcube]=laser_calcube2(laser_wl, cdata, handles);
    %laser_wl=534.3;
    load wave_disp_cf3.mat % hardware calibration data from Duke   
    
    % Find blue (right)
    blue=abs(round(disp_fit(start_wvl)-disp_fit(laser_wl)));
    % Red (left, subtract)
    red=-abs(round(disp_fit(end_wvl)-disp_fit(laser_wl)));

    
    % for color cube
    calcube=zeros(size(cdata,1),size(cdata,2),length(red:blue),'single');
    ind2=1;
    for ind=blue:-1:red % why is it -2?
        %calcube(:,:,ind2)=circshift(cdata,[0,ind]);% DUKE --> THIS IS WRONG DIRECTION!!!
        calcube(:,:,ind2)=circshift(cdata,[0,-ind]);% Min's try to change the dispersion direction
        ind2=ind2+1;
    end
    
    % wvls=linspace(425,750,size(calcube,3));
%    wvls=wvls_fit(disp_fit(550)-blue:disp_fit(550)-red); % by Duke
    wvls=wvls_fit(disp_fit(laser_wl)-blue:disp_fit(laser_wl)-red);% by Min
    %wvls=disp_fit(disp_fit(425)-(0:length(blue:red)-1) )+7;
    
    disp('Calcube Loaded...');
    disp(['Calcube: ' num2str(size(calcube,1)) 'x' num2str(size(calcube,2))...
        'x' num2str(size(calcube,3))]);
    
%{
    % this is from old CASSI
    shiftb=[-shift(1,:);shift(2,:)];
    % Since +movement is to the left, invert horizontal
    shiftb(1,:)=shiftb(1,:); % - with/after 9/10
    % % no need to invert vertical since lens does this:
    shiftb(2,:)=-shiftb(2,:);  % + prior to 9/8
    % Just in case round-off-errors:
    shiftb=round(shiftb);
%}    
    %=====================================================================
    % Min's CASSI
    % shift(1) is x and shift(2) is y
    % since I changed gig_simple.m as below:
    % (duke) rot90(handles.activex1.GetRawData,1);
    % (Min) fliplr(rot90(handles.activex1.GetRawData,1));
    % I have to change shift direction from left to right!
    % (1) spectral dispersion shift direction -> (Min's) calcube(:,:,ind2)=circshift(cdata,[0,-ind]);
    % (2) shift(1) direction: (duke)-shift(1,:) -> (Min's) shift(1,:);
    shiftb=[shift(1,:);-shift(2,:)];
    shiftb=round(shiftb);
    %=====================================================================
    % Spectral Subsampling for efficiency 
    % I think VIS and UV should be done differently.
    % we need to increase the number for VIS and decrease for UV.
    %
    % below part reduces spectral resolution
    ind2=1;
    wvls2=zeros(1,round(size(calcube,3)/deltab)); 
    % [Min] it should be single for memory efficiency (no benefit for
    % double), this is simple BW mask
    %Cu=zeros(size(calcube,1),size(calcube,2),round(size(calcube,3)/deltab)); % original
    Cu=zeros(size(calcube,1),size(calcube,2),round(size(calcube,3)/deltab), 'single'); % by Min
    for ind=1:deltab:size(calcube,3)
        wvls2(ind2)=wvls(ind);
        Cu(:,:,ind2)=calcube(:,:,ind);
        ind2=ind2+1;
    end

    % Reduce to specified wavelength:
    start_wvl=find(wvls2<=start_wvl); if ~isempty(start_wvl), start_wvl=start_wvl(end); else start_wvl=1; end
    end_wvl=find(wvls2>=end_wvl);  if ~isempty(end_wvl), end_wvl=end_wvl(1); else end_wvl=length(wvls2); end
    wvls2b=wvls2(start_wvl:end_wvl);
    Cu=Cu(:,:,start_wvl:end_wvl);

    disp('Done Building data set!');
    disp(['Cu: ' num2str(size(Cu,3))]);
    disp(['Start wv: ' num2str(wvls2(start_wvl))]);
    disp(['End wv: ' num2str(wvls2(end_wvl))]);
    disp(['Spectrum bins: ' num2str(floor(wvls2b))]);
    
    % After subsampling, delete the original 'calcube' and 'cdata'
    % for acquiring more memory space.
    clear ('calcube');
    clear ('cdata');

    % normalizing the inputs, Cu and I_trans
    Cu = single(Cu)./255.; % Cu -> Coded Aperture
    I_trans = single(I_trans)./255; % I_trans -> Shifted Images
    
    %=====================================================================
    % each piece calculation

    inum = height/piecedepth;
    tttime = tic;

%{
    % Spectral segmentation idea is totally wrong!
    %
    fnreconfilenamelist = [type '_reconfilelist.txt'];    
    if nargin<=1
        reconfilelist = fopen(fnreconfilenamelist,'w');
    else
        reconfilelist = fopen(fnreconfilenamelist,'a');
    end    
    
    % spectral band-width loop
    for j=1:size(Cu,3)
        disp(['Spectral Band Index: ' num2str(j)]); 
        disp(['Wavelength: ' num2str(floor(wvls2b(j))) 'nm']); 
        % one bandwidth of Cu
        CuB = zeros(size(Cu,1),size(Cu,2),1,'single');
        CuB(:,:,1) = Cu(:,:,j);
%}        
            
        % file list
        filenamelist = ['filelist_rec.txt'];
%        filenamelist = regexprep(filenamelist, '_rec.txt', ['_' num2str(floor(wvls2b(j))) 'nm_rec.txt'],'ignorecase');    
        
       
        if nargin<=1
            filelist = fopen(filenamelist,'w');
        else
            filelist = fopen(filenamelist,'a');
        end
        % spatial parallelization loop
        for i=starti:inum
            disp(['========================================']);
            disp(['Spatial Patch Index: ' num2str(i)]); 

            st = (i-1)*piecedepth + 1;
            ed = (i-1)*piecedepth + (piecedepth);

            % separating an image
            Cup = Cu(st:ed, :, :); % multi-spectral
%            Cup = CuB(st:ed, :, :); % one spectral band
            I_transp = I_trans(st:ed,:, :);

            % run iteration
%            piececalculation(i, sourcepath, filelist, Cup, I_transp,...
%            iterationsb, shiftb, taub, piterb, deltab, wvls2b(j),reconfilelist);
            piececalculation(i, sourcepath, filelist, Cup, I_transp,...
            iterationsb, shiftb, taub, piterb, deltab, wvls2b, tolA);
        end % spetial
        
        % close list file
        fclose(filelist); % it doesn't work (maybe matlab bug)

        % merge calculated data
%        merge_segments(type,filenamelist);
%{        
    end % spectral 
%}
    toc(tttime);
    disp('Total reconstruction finished');
%    disp(['Time: ' num2str(tttime)]); 

    fclose('all');        

    % merge individual calculation into one
    %merge_segments(type,filenamelist);
    merge_segments(type,filenamelist);
    %merge_channels(type,fnreconfilenamelist); % this is totally wrong

%{
    for j=1:size(Cu,3)
        disp(['Spectral Band Index: ' num2str(j)]); 
        disp(['Wavelength: ' num2str(floor(wvls2b(j))) 'nm']);   
        % file list
        %filenamelist = ['filelist_rec.txt'];
        %filenamelist = regexprep(filenamelist, '_rec.txt', ['_' num2str(floor(wvls2b)) 'nm_rec.txt'],'ignorecase');    
        
        % merge calculated data
        %merge_segments(type,filenamelist);
        

%         %==========================================================
%         % merge calculated data
%         if strcmp(type,'vis')==1 
%             merge_segments; %'filelist_rec.txt'
%         elseif strcmp(type,'uv')==1 
%             merge_segments_uv; %'filelist_rec.txt'
%         end   
%         %==========================================================
    end
%}  
end


function piececalculation(index, sourcepath, filelist, Cup, I_transp,...
    iterationsb, shiftb, taub, piterb, deltab, wvls2b, tolA)
%    iterationsb, shiftb, taub, piterb, deltab, wvls2b, reconfilelist)


    tic
    %[x_twist_orig obj_twist]=RUNME_calcuberecon2(Cu2, y2, iterationsb, shiftb, taub, piterb,rect2crop);
    [x_twist, obj_twist]=RUNME_calcuberecon_min(Cup, I_transp, iterationsb, shiftb, taub, piterb, tolA); % by Min
    %[x_twist obj_twist]= MKL_calcuberecon(Cup, I_transp, iterationsb, shiftb, taub, piterb, pathname); % by Hongzhi
    ttime=toc; 
      
    disp('Each reconstruction finished');
    disp(['Time: ' num2str(ttime)]);  
    
    %=====================================================================
    % displaying final => this could be wrong!
    % deltab decides the depth of spectral intervals.
    % This part can be used to solve the misalignment problem of spectral
    % wavelength
    
%{   
    cal2=zeros(size(x_twist),'single'); 
    xs=size(x_twist,2); ys=size(x_twist,1); mid=round(size(x_twist,3)/2);
    zs=size(x_twist,3);
    for ind=1:zs
        if deltab==1
            cal2(:,:,ind)=circshift(x_twist(:,:,ind),[0,(ind-1)-mid,0]);
        elseif deltab==2
            %cal2(:,:,ind)=circshift(x_twist(:,:,ind),[0,(ind-1)*2-mid*2,0]);
            cal2(:,:,ind)=circshift(x_twist(:,:,ind),[0 -((zs-1)*2-(ind-1)*2)+zs]);% This is WRONG!
        else
            cal2=x_twist;
        end
    end
    %cal2=circshift(cal2,[0 zs]);
    x_twist = cal2;
%}
    
    %=====================================================================
    % Save data as a file
    %savefilename = regexprep(sourcepath, '.mat', ['_' num2str(floor(wvls2b)) 'nm_' num2str(index) '_rec.mat'],'ignorecase');    
    savefilename = regexprep(sourcepath, '.mat', ['_' num2str(index) '_rec.mat'],'ignorecase');    

   
    save(savefilename,'shiftb','taub','piterb','Cup', 'iterationsb',...
        'wvls2b','deltab','x_twist','ttime')

    % write the filename into main list file
%    fprintf(reconfilelist, '%s\n', savefilename); % write filename   
    
    
    disp('Done saving data');      

    % save filename
    fprintf(filelist, '%s\n', savefilename); % write filename    
end

function [x_twist_orig obj_twist]= MKL_calcuberecon(Cu, cdata, its, shift,tau,piter, pathname)% edited by Min
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
write_matrix_bin([pathname 'Cu.bin'], Cu);
write_matrix_bin([pathname 'cdata.bin'], cdata);
write_matrix_bin([pathname 'shift.bin'], shift);
write_param([pathname 'param.txt'], its, tau, piter);

%!C:\Scanning_System\ImagingSystem\CPUAccelerated on Code_Feb_2010\Simulation\twist
%!C:\Scanning_System\Cassi_Imaging_Pipeline\Cassi_Program\rec_gui\c_code\twist\Debug\twist
%!C:\Scanning_System\Cassi_Imaging_Pipeline\Cassi_Program\rec_gui\c_code\twist\Release\twist
!twist.exe

x_twist_orig = read_matrix_bin([pathname 'x_twist.bin']);
obj_twist = read_matrix_bin([pathname 'obj_twist.bin']);

end

% close all;clear all;clc;
function [x_twist_orig obj_twist]=RUNME_calcuberecon_min(Cu, cdata, its, shift,tau,piter,tolA)

% rot=zeros(size(shift));

% Image to process:
%y = double(cdata); % cdata is just BW floating data
y = cdata; % single precision should be enough
clear('cdata');

y = y.*(y>=0);
%y=reshape(y,[size(y,1),size(y,2),size(shift,2)]); % (Min) we don't need
%this line (technically no change on the matrix)

%y = flipdim(y,1);

n1 = size(y,1);
n2 = size(y,2);
nt=size(shift,2);
m = size(Cu,3);
% y = y/max(y(:)); 
y=y(:);

% Single precision:
%y=single(y);
%Cu=single(Cu);

% (Min) normalization? why? necessary? -> by test, without normalization
% this TWIST algorithm failed. So we have to fix this MAXIMUM y and Cu
% consistently.
%y=y/(max(y(:))*1);
%Cu=Cu/(max(Cu(:))*1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% shifting measurement cube and building up 4D measurement cube data.
%tic
Cus=zeros(size(Cu,1),size(Cu,2),m,nt);
for ind=1:nt
    Cint=circshift(Cu,[shift(2,ind),shift(1,ind),0]);
    Cus(:,:,:,ind)=Cint(1:size(Cu,1),1:size(Cu,2),:);
end
%toc


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% smoothing parameter (empirical)
% tau = .3;

% ------------  TV experiments ---------------------
% handle functions for TwIST
%  convolution operators

% A = @(x) Rfuntwistv2(x,n1,n2,m,Cu,shift,rot);
% AT = @(x) RTfuntwistv2(x,n1,n2,m,Cu,shift,rot);
A = @(f) R2(f,n1,n2,m,Cus,nt);
AT = @(y) RT2(y,n1,n2,m,Cus,nt);

% denoising function;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  New TV denoising code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% piter = 4;

% Psi = @(x,th) MyTVpsi(x,th,tau,piter,n1,n2,m,[1 1 0]);
Psi = @(x,th) cassidenoise(x,th,piter); % from Duke
%Psi = @(x,th) tvdenoise(x,th,piter); % this function didn't work well (for reconstruction)

% Phi = @(x) MyTVphi(x,n1,n2,m,[1 1 0]);
% Phi = @(x) TVnormspectralimaging(x);
Phi = @(x) TVnorm3D(x);

%tolA = 1e-8; %DUKE

% -- TwIST ---------------------------
% stop criterium:  the relative change in the objective function
% falls below 'ToleranceA'

[x_twist_orig,dummy,obj_twist,times_twist,dummy,mse_twist]=TwIST( ...
    y,A,tau,...
    'AT', AT, ...
    'Psi', Psi, ...
    'Phi',Phi, ...
    'Initialization',2,...
    'Monotone',1,...
    'StopCriterion',1,...
    'MaxIterA',its,...
    'ToleranceA',tolA,...
    'Debias',0,...
    'Verbose', 1);

temp = shiftCube(x_twist_orig);
x_twist_orig = temp; 


% x_twist = flipdim(x_twist_orig,1);
% 
% x_twist = x_twist.*(x_twist>=0);
% x_twist = x_twist/max(x_twist(:));
% 
% dispCubeAshwin(x_twist,140,round(linspace(452,660,25)));


% x_twist = flipdim(x_twist_orig,1); %x_twist_orig; 
% temp = shiftCube(x_twist);
% x_twist = temp; %temp(:,end-247:end,:);
% x_twist = x_twist.*(x_twist>=0);
% x_twist = x_twist/max(x_twist(:));
% 
% dispCubeAshwin(x_twist,200,round(linspace(453,653,z_num)));

% figure(1000);
% plot(log(obj_twist),'r','LineWidth',2,'LineWidth',2)
% legend('TwIST')
% st=sprintf('tau = %2.2e',tau);  title(st)
% ylabel('Obj. function')
% xlabel('CPU time (sec)')
% grid on

% toc
end




function y = TVnorm3D(x)
% this is phi function (this produces the summation of the magnitudes of gradients)
% TVnonmspectralimging --> one constant
nt=size(x,3);
y=zeros(1,nt,'single');
z1=zeros(size(x,1),1,'single');
z2=zeros(1,size(x,2),'single');

%  the summation of the magnitudes of gradients
for ind = 1:nt
    y(ind) = sum(sum(sqrt(diffh(x(:,:,ind),z1).^2+diffv(x(:,:,ind),z2).^2)));
end

y = sum(y);

end

% h=[0 1 -1];
function y=diffh(x,z1)
% z1=zeros(size(x,1),1);
x = [x z1]-[z1 x];
x(:,1) = x(:,end)+x(:,1);
y = x(:,1:end-1);
end

% h=[0 1 -1]';
function y=diffv(x,z2)
% z2=zeros(1,size(x,2));
x = [x; z2]-[z2; x];
x(1,:) = x(end,:)+x(1,:);
y = x(1:end-1,:);
end

% For normal cpu:
function y=R2(f,n1,n2,m,Cs,nt) % y = Ax (h*w*snap)
% Number of images:
%nt=size(shift,2);

% Make sure in 4D shape:
f=reshape(f,[n1,n2,m]);

% Punch:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Min] Here multidimensional dot product was impremented 
%       as elementary product plus sum 
%       1: h, 2: w, 3: spectral channels, 4: aperture snaps
% Elementary product
gp=repmat(f,[1 1 1 nt]).*Cs; % 4D * 4D
%gp=bsxfun(@times, f,Cs);
% Sum up the 3rd dimensions in each seperate image:
y=sum(gp,3); % 4D -> 3D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y=y(:); % vectorize
end

function f=RT2(y,n1,n2,m,Cs,nt) % f = ATy (h*w*spec)
% this is for captured image (y) and Cs is coded aperture (Cu)

% nt=size(shift,2);
% Make sure it is in 4D shape
% n1=rows, ie 480
% n2=cols, ie 640
% length(shift) = number of individual images
y=reshape(y,[n1,n2,1,nt]);

% Replicate to number of channels in system:
yp=repmat(y,[1,1,m,1]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Min] Here multidimensional dot product was impremented 
%       as elementary product plus sum 
%       1: h, 2: w, 3: spectral channels, 4: aperture snaps
% Elementary product
yp=yp.*Cs; % 4D * 4D
% Sum up on the 4th dimension:
f=sum(yp,4); % 4D -> 3D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
