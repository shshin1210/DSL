function [fitresult, gof] = createFit(x, y, xo)
%CREATEFIT(X,Y,XO)
%  피팅을 생성하십시오.
%
%  'untitled fit 1' 피팅의 데이터:
%      X 입력값: x
%      Y 입력값: y
%      Z 출력값: xo
%  출력값:
%      fitresult: 피팅을 나타내는 피팅 객체.
%      gof: 피팅의 적합도 정보를 포함한 구조체.
%
%  참고 항목 FIT, CFIT, SFIT.

%  MATLAB에서 16-Aug-2023 17:17:53에 자동 생성됨


%% 피팅: 'untitled fit 1'.
[xData, yData, zData] = prepareSurfaceData( x, y, xo );

% fittype과 옵션을 설정하십시오.
ft = fittype( 'p00+p10*x+p01*y+p20*x^2+p11*x*y+p02*y^2+p30*x^3+p21*x^2*y+p12*x*y^2+p03*y^3+p40*x^4+p31*x^3*y+p22*x^2*y^2+p13*x*y^3+p04*y^4+p50*x^5+p41*x^4*y+p32*x^3*y^2+p23*x^2*y^3+p14*x*y^4+p05*y^5+p60*x^6+p51*x^5*y+p42*x^4*y^2+p33*x^3*y^3+p24*x^2*y^4+p15*x*y^5+p06*y^6', 'independent', {'x', 'y'}, 'dependent', 'z' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [0.133503859661312 0.021555887203497 0.55984070587251 0.300819018069489 0.939409713873458 0.980903636046859 0.286620388894259 0.800820286951535 0.896111351432604 0.59752657681783 0.88401673572382 0.943731541195791 0.549158087419903 0.728386824594357 0.57675829785801 0.0258574710831396 0.446530978284814 0.646301957350656 0.521202952672418 0.372312660779512 0.937134666341562 0.829532824526515 0.849085479954455 0.372534239899539 0.59318457521849 0.872552564647559 0.933501608507105 0.668464274361376];

% 데이터에 모델을 피팅하십시오.
[fitresult, gof] = fit( [xData, yData], zData, ft, opts );

% 데이터의 피팅을 플로팅하십시오.
figure( 'Name', 'untitled fit 1' );
h = plot( fitresult, [xData, yData], zData );
legend( h, 'untitled fit 1', 'xo vs. x, y', 'Location', 'NorthEast', 'Interpreter', 'none' );
% 좌표축에 레이블을 지정하십시오.
xlabel( 'x', 'Interpreter', 'none' );
ylabel( 'y', 'Interpreter', 'none' );
zlabel( 'xo', 'Interpreter', 'none' );
grid on
view( -12.3, 23.6 );


