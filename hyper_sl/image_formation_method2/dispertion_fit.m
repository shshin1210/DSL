function [fitresult, gof] = dispertion_fit_method2(x, y, xo)
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

%  MATLAB에서 08-Aug-2023 17:16:45에 자동 생성됨


%% 피팅: 'untitled fit 1'.
[xData, yData, zData] = prepareSurfaceData( x, y, xo );

% fittype과 옵션을 설정하십시오.
ft = fittype( 'poly55' );

% 데이터에 모델을 피팅하십시오.
[fitresult, gof] = fit( [xData, yData], zData, ft );

% 데이터의 피팅을 플로팅하십시오.
% figure( 'Name', 'test fit' );
% h = plot( fitresult, [xData, yData], zData );
% legend( h, 'test fit', 'xo vs. x, y', 'Location', 'NorthEast', 'Interpreter', 'none' );
% % 좌표축에 레이블을 지정하십시오.
% xlabel( 'x', 'Interpreter', 'none' );
% ylabel( 'y', 'Interpreter', 'none' );
% zlabel( 'xo', 'Interpreter', 'none' );
% grid on
% view( -13.4, -11.2 );


