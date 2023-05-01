

function [ hs_img ] = vccassi( filename_coded_A,...
                                            filename_cassi_img,...
                                            crect,...
                                            start_wvl,...
                                            end_wvl,...
                                            laser_wl,...
                                            deltab )
%% Init
if nargin == 0
    filename_coded_A = '';
    filename_cassi_img = '';
    laser_wl = 550;
    deltab = 6;
    start_wvl = 450;
    end_wvl = 720;
end
IS_VCCASSI = true;

%% Algorithm param setting
%============================================%
%     Parameters for reconstruction          %
%============================================%
%------------------------------------------------------------------
% denoising calculatin (smoothing factor parameters) in a single
% iteration.
%------------------------------------------------------------------
taub = 0.001;
%------------------------------------------------------------------
% piterb is a number of iteration in a single denoise operation.
% for tvdenoise.m fuction
%------------------------------------------------------------------
piterb=4;
%------------------------------------------------------------------
% Maximum TWIST iteration (from Ver. 5)
%------------------------------------------------------------------
iterationsb=200; % reasonable maximum
%------------------------------------------------------------------
% Iteration stop criteria in TWIST: tolerance to stop iterations.
%------------------------------------------------------------------
 % this is important to keep for iterating higher number (unlimited iteration) -> make no sense
tolA = 1e-8;

%% Read coded mask
if( isempty( filename_coded_A ) )
    [ filename_coded_A, filepath_coded_A]...
        = uigetfile( { '*.mat','mat';  }, 'Choose a calibration file' );
    
    if size( filepath_coded_A, 2 ) == 1
        return;
    end
else
    filepath_coded_A = [pwd '/'];
end
source_calib = fullfile( filepath_coded_A, filename_coded_A );
load( source_calib );       % This will load coded_A, of which size is height*width*#images

% read the size of cdata
height = size( coded_A, 1 );
width = size( coded_A, 2 );
n_coded_As = size( coded_A, 3 );
disp 'Calibration data reading done!';



%% Read cassi image
% ask filename for the main listfile
if( isempty( filename_cassi_img ) )
    % file read
    [ filename_cassi_img, filepath_cassi_img ]...
        = uigetfile( {'*.mat','mat'; }, 'Choose a main capture mat file' );
    % leave function when no file is selected.
    if size( filepath_cassi_img,2 ) == 1
        return;
    end
else
    filepath_cassi_img = [pwd '/'];
end
sourcepath = fullfile( filepath_cassi_img, filename_cassi_img );
% load the source MAT file( I_trans )
load( sourcepath );



%% Select Region
img_i = uint8( I_trans( :, :, 1 ) );
[ X crect ] = selectregion( img_i );

regionheight = crect(4)+1;
regionwidth = crect(3)+1;
regiondepth = size( I_trans, 3 );
regionvolume = regionheight * regionwidth * regiondepth;

coded_A...
    = coded_A ( crect(2):crect(2) + crect(4), crect(1):crect(1) + crect(3), :);
I_trans...
    = I_trans( crect(2):crect(2) + crect(4), crect(1):crect(1) + crect(3), : );
clear('img_i', 'X');
disp(['========================================']);
disp([sprintf( '[Selected region in %dx%d]', height, width )]);
disp(['Rect( x, y, w-1, h-1 ):' num2str( uint16( crect ) ) ] );
% show cropped results
for i = 1:n_coded_As
    figure; imshow( uint8( coded_A( :,:, i ) ) ), title( sprintf( '%d th coded mask', i ) ); 
end
figure; imshow( uint8( I_trans( :, :, 1 ) ) );set(gcf,'name','Dispersion capture');

%% Displaying Params
disp(['========================================']);
disp(['[IMAGE_SIZE]']);
disp(['Height: ' num2str( height ) ] );
disp(['Width: ' num2str( width ) ] );
disp(['']);
disp(['[VARIABLES]']);
% disp(['Spectral Type: ' type]);
%    disp(['Piece Depth: ' num2str(piecedepth) 'px']);
disp(['taub: ' num2str(taub)]);
disp(['piterb: ' num2str(piterb)]);
disp(['iterationsb: ' num2str(iterationsb)]);
disp(['tolA: ' num2str(tolA)]);
disp(['deltab: ' num2str( deltab ) ] );
disp(['start wvl: ' num2str( start_wvl ) ] );
disp(['end wvl: ' num2str( end_wvl ) ]);
% disp(['starting index: ' num2str(starti)]);
disp(['========================================']);
disp(['[START]']);

%% Sample the wavelentgh
% load dispersion calibration data
if IS_VCCASSI
    load vccassi_calib_data.mat
else
    load wave_disp_cf3.mat % hardware calibration data from Duke   
end
% Find blue (right)
blue = abs( round( disp_fit( start_wvl ) - disp_fit( laser_wl ) ) );
% Red (left, subtract)
red = -abs( round( disp_fit( end_wvl ) - disp_fit( laser_wl ) ) );

% for color cube
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calcube = zeros( size(coded_A,1), size(coded_A,2), length(red:blue), 'single');
% ind2=1;
% for ind=blue:-1:red % why is it -2?
%     calcube( :, :, ind2 ) = coded_A;
%     ind2=ind2+1;
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
calcube = zeros( regionheight, regionwidth, length(red:blue), n_coded_As, 'single');
ind2 = 1;
for ind = blue:-1:red % why is it -2?
    for iter_mask = 1:n_coded_As
        calcube( :, :, ind2, iter_mask ) = coded_A( :, :, iter_mask );
    end
    ind2 = ind2 + 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
wvls = wvls_fit(disp_fit(laser_wl)-blue:disp_fit(laser_wl)-red);% by Min
disp('Calcube Loaded...');
disp(['Calcube: ' num2str( regionheight ) 'x' num2str( regionwidth )...
    'x' num2str(size(calcube,3)) 'x' num2str( n_coded_As )]);

% images are already aligned.
% There is no shift between coded patterns, but they are just different.
% shiftb = [ shift(1,:); -shift(2,:) ]; % case1: current version (12/28/2011)
% shiftb = round(shiftb); % case1: current version (12/28/2011)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ind2=1;
% wvls2 = zeros(1,round(size(calcube,3)/deltab),'single');
% Cu = zeros(size(calcube,1),size(calcube,2),round(size(calcube,3)/deltab), 'single'); % by Min
% for ind=1:deltab:size(calcube,3)
%     wvls2(ind2)=wvls(ind);
%     Cu(:,:,ind2)=calcube(:,:,ind);
%     ind2=ind2+1;
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ind2 = 1;
wvls2 = zeros( 1, round( size( calcube,3 )/deltab ),'single' );
Cu = zeros( size( calcube, 1 ),size( calcube, 2 ),...
        round(size(calcube,3)/deltab), size( calcube, 4 ), 'single');
for ind = 1:deltab:size(calcube,3)
    wvls2(ind2)=wvls(ind);
    for iter_mask = 1:n_coded_As
        Cu( :, :, ind2, iter_mask ) = calcube( :, :, ind, iter_mask );
    end
    ind2=ind2+1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

start_wvl=find(wvls2<=start_wvl);
if ~isempty(start_wvl), start_wvl=start_wvl(end); else start_wvl=1; end
end_wvl=find(wvls2>=end_wvl);
if ~isempty(end_wvl), end_wvl=end_wvl(1); else end_wvl=length(wvls2); end
wvls2b=wvls2(start_wvl:end_wvl);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cu=Cu(:,:,start_wvl:end_wvl);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Cu = Cu( :, :, start_wvl:end_wvl, :);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disperse = int32(round( disp_fit( laser_wl ) -  disp_fit( wvls2b ) ));

disp('Done Building data set!');
disp(['Cu (# of Channels): ' num2str(size(Cu,3))]);
disp(['Start wv: ' num2str(wvls2(start_wvl))]);
disp(['End wv: ' num2str(wvls2(end_wvl))]);
disp(['Spectrum bins: ' num2str(floor(wvls2b))]);

clear ('calcube');
clear ('coded_A');

%% Do Recon
Cu = single(Cu)./255.; % Cu -> Coded Aperture
I_trans = single(I_trans)./255.; % I_trans -> Shifted Images

hs_img = piececalculation(1, sourcepath, Cu, I_trans,...
                         disperse, n_coded_As, iterationsb,...
                         taub, piterb, deltab, wvls2b, tolA);
end



function [ hs_img ] = piececalculation( index, sourcepath, Cup, I_transp,...
    disperse, n_coded_As, iterationsb, taub, piterb, deltab, wvls2b, tolA )


idttime = tic;
[x_twist] = RUNME_calcuberecon_min(Cup, I_transp, disperse, n_coded_As, iterationsb, taub, piterb, tolA);
hs_img = x_twist;
ttime=toc(idttime);

disp('Reconstruction of a piece was finished');
disp(['Time: ' formattime(ttime)]);

[pathstr, name, ext] = fileparts(sourcepath);
name = regexprep(name, 'merged_', ['recon_' num2str(index) '_']);
savefilename = [pathstr, '/', name, ext];

save(savefilename,'taub','piterb','Cup', 'iterationsb',...
    'wvls2b','deltab','x_twist','ttime')

disp('Done saving data');

end

function [x_twist_orig] = RUNME_calcuberecon_min(Cu, cdata, disperse, n_coded_As, its,tau,piter,tolA)
% Cu: Transformation tensor
% cdata: Sensored Images (W x H x n)
% its: Iteration number

% Image to process:
y = cdata; % single precision should be enough
clear('cdata');

y = y.*(y>=0);
n1 = size(y,1);
n2 = size(y,2);
% nt = size(shift,2);
nt = n_coded_As;
m = size( Cu, 3 );


% shifting measurement cube and building up 4D measurement cube data.
%tic

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cus = zeros(size(Cu,1),size(Cu,2),m,nt,'single');
% for ind=1:nt
%     Cint = circshift(Cu,[shift(2,ind),shift(1,ind),0]);
%     Cus(:,:,:,ind) = Cint(1:size(Cu,1),1:size(Cu,2),:);
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Cus = Cu;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


A = @(f) R2(f,n1,n2,m,Cus,nt, disperse);
AT = @(y) RT2(y,n1,n2,m,Cus,nt, disperse);
Psi = @(x,th) cassidenoise( x, th, piter ); % from Duke
Phi = @(x) TVnorm3D(x);


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
    'Verbose', 1 );

end


function y = TVnorm3D(x)
% this is phi function (this produces the summation of the magnitudes of gradients)
% TVnonmspectralimging --> one constant
nt=size(x,3);
y=zeros(1,nt,'single');
z1=zeros(size(x,1),1,'single');
z2=zeros(1,size(x,2),'single');

%  the summation of (the magnitudes of gradients).^2
for ind = 1:nt
% original cassi: the summation of the magnitudes of gradients
%     y(ind) = sum(sum(sqrt(diffh(x(:,:,ind),z1).^2+diffv(x(:,:,ind),z2).^2)));
% revised by Min (the summation of the magnitudes of gradients)^2 
% on 4/5/2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [INCHANG] Use L1 norm!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   y(ind) = sum(sum((abs(diffh(x(:,:,ind),z1))+abs(diffv(x(:,:,ind),z2)))));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    y(ind) = sum(sum((diffh(x(:,:,ind),z1).^2+diffv(x(:,:,ind),z2).^2)));
% for some reason, it didn't work properly with some images.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%[INCHANG]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = R2( f, n1, n2, m, Cs, nt, disperse) % y = Ax (h*w*snap)
% function y = R2( f, n1, n2, m, Cs, nt) % y = Ax (h*w*snap)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OLD R2() implementation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Make sure in 4D shape:
% f=reshape(f,[n1,n2,m]);
% % Punch:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % [Min] Here multidimensional dot product was impremented 
% %       as elementary product plus sum 
% %       1: h, 2: w, 3: spectral channels, 4: aperture snaps
% % Elementary product
% gp=repmat(f,[1 1 1 nt]).*Cs; % 4D * 4D
% % Sum up the 3rd dimensions in each seperate image:
% y=sum(gp,3); % 4D -> 3D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% y=y(:); % vectorize

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [INCHANG] New R2 implementation by INCHANG
% (1) Disperse the hyperspectral channels after masking the coded aperture
% (2) Shift the hyperspectral image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vector to Image
f = reshape( f, [ n1, n2, m ] );

% Apply masking by the coded aperture
gp = repmat( f, [1 1 1 nt] ).*Cs;

% Disperse the masked channels
for i = 1:m
    gp_1channels = gp( :, :, i, : );
    if isempty( disperse )
        gp( :, :, i, : ) = circshift( gp_1channels, [ 0, -i, 0 ] );
    else
        if nt == 5
              gp( :, :, i, 1 ) = circshift( gp_1channels(:,:,1,1), [ 0, -disperse( i ), 0 ] );
              gp( :, :, i, 5 ) = circshift( gp_1channels(:,:,1,5), [ 0, -disperse( i ), 0 ] );
              gp( :, :, i, 2:4 ) = circshift( gp_1channels(:,:,1,2:4), [ 0, disperse( i ), 0 ] );
        else
            gp( :, :, i, : ) = circshift( gp_1channels, [ 0, disperse( i ) , 0 ] ); % (Min) blue must be right
        end
    end
end
% Projection
y = squeeze( sum( gp, 3 ) );
% Vectorize
% y = y(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

%[INCHANG]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = RT2( y, n1, n2, m, Cs, nt, disperse ) % f = ATy (h*w*spec)
% function f = RT2( y, n1, n2, m, Cs, nt ) % f = ATy (h*w*spec)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OLD RT2() implementation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % this is for captured image (y) and Cs is coded aperture (Cu)
% % nt=size(shift,2);
% % Make sure it is in 4D shape
% % n1=rows, ie 480
% % n2=cols, ie 640
% % length(shift) = number of individual images
% y=reshape(y,[n1,n2,1,nt]);
% % Replicate to number of channels in system:
% yp=repmat(y,[1,1,m,1]);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % [Min] Here multidimensional dot product was impremented 
% %       as elementary product plus sum 
% %       1: h, 2: w, 3: spectral channels, 4: aperture snaps
% % Elementary product
% yp=yp.*Cs; % 4D * 4D
% % Sum up on the 4th dimension:
% f=sum(yp,4); % DUKE: 4D -> 3D
% f = f(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [INCHANG] New RT2 implementation by INCHANG
% (1) Disperse the hyperspectral channels after masking the coded aperture
% (2) Shift the hyperspectral image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vector to Image
y = reshape( y, [ n1, n2, 1, nt ] );
% y = reshape( y, [ n2, n1, 1, nt ] );
% Replicate
yp = repmat ( y, [ 1, 1, m, 1 ]);
% disperse
for i = 1:int32( m )
%     gp(:,:,i) = circshift( gp, [ 0, -i ] );
    yp_1channels = yp( :, :, i, : );
    if isempty( disperse )
        yp( :, :, i, : ) = circshift( yp_1channels, [ 0, i, 0 ] );
    else
        if nt == 5
            yp( :, :, i, 1 ) = circshift( yp_1channels(:,:,1,1), [ 0, disperse( i ), 0 ] );
            yp( :, :, i, 5 ) = circshift( yp_1channels(:,:,1,5), [ 0, disperse( i ), 0 ] );
            yp( :, :, i, 2:4 ) = circshift( yp_1channels(:,:,1,2:4), [ 0, -disperse( i ), 0 ] );
        else
            yp( :, :, i, : ) = circshift( yp_1channels, [ 0, -disperse( i ), 0 ] ); % (Min) the inverse of blue must be the inverse of right
        end
        
    end
end
% Mask
yp = yp.*Cs;   
% Proejction
f = sum( yp, 4 ); % DUKE: 4D -> 3D
% Vectorize
% f = f(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end