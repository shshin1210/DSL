function XYZ=D(K)
% I change the white point values to get it more accurate
% by reference of CIE 15.3 and ISO13655:1996 and ASTM E308-95
% D: returns XYZ tristimulus values of various CIE illuminants.
%
%        xyz=D(50) returns XYZ values for the D50 illuminant
%        xyz=D(55) returns XYZ values for the D55 illuminant
%        xyz=D(65) returns XYZ values for the D65 illuminant
%        xyz=D(75) returns XYZ values for the D75 illuminant
%        xyz=D('A') returns XYZ values for Illuminant A
%        xyz=D('C') returns XYZ values for Illuminant C
%
% 'D' with no arguments displays the tristimulus values of 
% all four D illuminants together with A and C illuminants.
%
% XYZ=D displays the values as above and returns the array
% of tristimulus values for all the illuminants
%
% old version of Phil Green
%   A=[109.85,100,35.8];
%   C=[98.07,100,118.23];
%   D50=[96.42,100,82.49];
%   D55=[96.68,100,92.14];
%   D65=[95.04,100,108.89];
%   D75=[94.96,100,122.62];
%   allXYZ=[A;C;D50;D55;D65;D75];
%
% Revised version by CIE 15.3

   A=[109.850,100,35.585];
   C=[98.074,100,118.232];
   D50=[96.422,100,82.521];
   D55=[95.682,100,92.149];
   D65=[95.047,100,108.883];
   D75=[94.96,100,122.62]; % I don't know, so it is not changed from before.
   allXYZ=[A;C;D50;D55;D65;D75];
   
   xyz=num2str(allXYZ,'%8.2f');
   labels=[' A    ';' C    ';' D50  ';' D55  ';' D65  ';' D75  '];
   all=[labels,xyz];

if nargin>0
   switch K
   case 'cie_a';XYZ=A;
   case 'cie_c';XYZ=C;
   case 'apple63'; XYZ=apple63;
   case 'apple60'; XYZ=apple60;
   case 50;XYZ=D50;
   case 55;XYZ=D55;
   case 60;XYZ=D60;
   case 65;XYZ=D65;
   case 75;XYZ=D75;
   end
else
   disp(all);
   if nargout>0;XYZ=allXYZ;end
end   

