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
% function B2 = apply_3dis_calib(A)
function B2 = apply_3dis_calib(A)
% radiometric calibration for 3DIS (53x53 matrix)

% 2011/11/21 for SIGGRAPH 2012 (ki)
%{  
RESULTS FOR ABS RADIANCE
-----------------------------------------------
Characterization result
<TRAINING>  --> REPORT
RSQ = 0.9867
meddiff = 12.1473
stdv = 0.0215
cv1 = 17.1438

<TEST> --> NOT MEANINGFUL DUE TO EXPOSURE SETTING CHANGE
RSQ = 0.9879
meddiff = 38.2346
stdv = 0.0233
cv1 = 47.5081
%}
%{  
RESULTS FOR REFLECTANCE
-----------------------------------------------
Characterization result
<TRAINING> --> NOT MEANINGFUL
RSQ = 0.9508
meddiff = 11.9792
stdv = 0.2561
cv1 = 21.4338

<TEST> ---> REPORT
RSQ = 0.9758
meddiff = 8.4450
stdv = 0.2960
cv1 = 13.4018
%}
ki = [0.887456469462613;1.24940853810834;0.841808307691339;0.922769495573361;0.811081358359964;0.736331007852257;0.649457203184891;0.548557055073548;0.463048012472187;0.383066585337466;0.294092577794653;0.222767046094377;0.160028693813040;0.154175158279621;0.154627573509406;0.187517134356074;0.384400206404295;0.369711148421207;0.367236250064422;0.367252033313616;0.352354500291086;0.334224137066259;0.318201426663793;0.316032554248975;0.320038594989011;0.327356913050982;0.337839256901789;0.357237795896838;0.384974641701413;0.391525833520315;0.399184707551480;0.429886134840936;0.408218712753231;0.384242508422197;0.549558329723681;0.449693091086760;0.593058186387947;0.658975181469759;0.653980265996210;0.640916980774153;0.766738755598117;1.32019057556633;0.751131565059433;3.47750929295767;0.600098778145124;0.476241850311746;1.50681640366731;1.50681640366731;1.50681640366731;1.50681640366731;1.50681640366731;1.50681640366731;1.50681640366731;];
%-----------------------------------------------
% [2012-01-13] 
% MIN: while I measuring UV fluorescence with a new light
% I found an overfitting at a wavelength of 825nm (where, Xenon shows a
% strong spick). 
% I think, this scalor is overfitted to the training samples.
% I would suggest to correct this manually, by using this measurement.
%-----------------------------------------------
% Overfitting correction for the spiky wavelength 825nm
% The previous scalar at ki(44), 825nm, should be changed for accuracy.
%
% ki fitting (before correction) under a different light (UV fluorecent)
% 807.9419	0.005885263	0.001075932	0.005155805	0.000996412	0.003522208
% 824.6056	0.004933432	0.000990163	0.003342736	0.000924333	0.002284471
% 841.8367	0.003315117	0.000550861	0.002350962	0.000519156	0.002432974

% ki fitting (after correction) under a different light (UV fluorecent)
% 807.9419	0.005885263	0.001075932	0.005155805	0.000996412	0.003522208
% 807.9419	0.00460019	0.000813396	0.003753384	0.000757784	0.002977591 % this is the avergage of the 825nm and 859nm (the original calibration under the xenon). --> This is overfitting.
% 841.8367	0.003315117	0.000550861	0.002350962	0.000519156	0.002432974

% The ratio of the overfiting				
% 0.104939791	0.092450504	0.126367231	0.092263629	0.146687453
% MEAN	STDEV
% 0.1125	0.0236
ki(44) = ki(44) * 0.1125;
%-----------------------------------------------

disp 'Applying radiometric calibration...';

h = size(A,1);
w = size(A,2);
d = size(A,3);
A = reshape(A, [h*w, d]);

kk = repmat(ki',[h*w, 1]);
% apply spectral linearization calibration (this is the hadama product)
B = A.*kk;

B2 = reshape(B, [h, w, d]);

% accounting for noise floor
B2(B2<9.6818e-06) = 9.6818e-06;

disp 'Done.';


% After applying Linearization
% 
% r21 =
% 
%     0.9867
% 
% 
% cv1 =
% 
%    17.1438
% 
% After applying Characterization on training
% 
% r21 =
% 
%     0.9952
% 
% 
% cv1 =
% 
%     6.3203
% 
% After applying Characterization on test
% 
% r21 =
% 
%     0.9803
% 
% 
% cv1 =
% 
%    11.8966
