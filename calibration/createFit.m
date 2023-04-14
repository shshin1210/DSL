function fitresult = createFit(x, y)
%CREATEFIT(B_X,B_Y)
%  피팅을 생성하십시오.
%
%  'b_curve' 피팅의 데이터:
%      X 입력값: b_x
%      Y 출력값: b_y
%  출력값:
%      fitresult: 피팅을 나타내는 피팅 객체.
%      gof: 피팅의 적합도 정보를 포함한 구조체.
%
%  참고 항목 FIT, CFIT, SFIT.

%  MATLAB에서 17-Nov-2022 14:46:25에 자동 생성됨


%% 피팅: 'rgb_curve'.
[xData, yData] = prepareCurveData( x, y );

% fittype과 옵션을 설정하십시오.
ft = fittype( 'a*(x)^b+c', 'independent', 'x', 'dependent', 'y' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.Lower = [-10 -10 -Inf];
opts.StartPoint = [0.340385726666133 0.585267750979777 0.223811939491137];
opts.Upper = [10 Inf Inf];

% 데이터에 모델을 피팅하십시오.
[fitresult, gof] = fit( xData, yData, ft, opts );

% 데이터의 피팅을 플로팅하십시오.
figure( 'Name', 'b_curve' );
h = plot( fitresult, xData, yData );
legend( h, 'y vs. x', 'b_curve', 'Location', 'NorthEast', 'Interpreter', 'none' );
% 좌표축에 레이블을 지정하십시오.
xlabel( 'x', 'Interpreter', 'none' );
ylabel( 'y', 'Interpreter', 'none' );
grid on



end