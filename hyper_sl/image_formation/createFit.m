function [fitresult, gof] = createFit(x, y)
%CREATEFIT(X,Y)
%  피팅을 생성하십시오.
%
%  'untitled fit 1' 피팅의 데이터:
%      X 입력값: x
%      Y 출력값: y
%  출력값:
%      fitresult: 피팅을 나타내는 피팅 객체.
%      gof: 피팅의 적합도 정보를 포함한 구조체.
%
%  참고 항목 FIT, CFIT, SFIT.

%  MATLAB에서 25-Oct-2023 21:55:30에 자동 생성됨


%% 피팅: 'untitled fit 1'.
[xData, yData] = prepareCurveData( x, y );

% fittype과 옵션을 설정하십시오.
ft = fittype( 'power2' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';

% 데이터에 모델을 피팅하십시오.
[fitresult, gof] = fit( xData, yData, ft, opts );

% 데이터의 피팅을 플로팅하십시오.
figure( 'Name', 'untitled fit 1' );
h = plot( fitresult, xData, yData );
legend( h, 'y vs. x', 'Depth fitting', 'Location', 'NorthEast', 'Interpreter', 'none' );
% 좌표축에 레이블을 지정하십시오.
xlabel( 'x', 'Interpreter', 'none' );
ylabel( 'y', 'Interpreter', 'none' );
grid on

if (gof.rmse > 1.)
    if(y(1) == 0)
        newy = y(2:end);
        newy(end) = y(end) + 1;
        newx = x(2:end);
    else
        newy = y(1:end);
        newy(end) = y(end) + 1;
        newx = x(1:end);
    end

    [xData, yData] = prepareCurveData( newx, newy );
    % fittype과 옵션을 설정하십시오.
    ft = fittype( 'power2' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    
    % 데이터에 모델을 피팅하십시오.
    [fitresult, gof] = fit( xData, yData, ft, opts );
end

if (gof.rmse > 1.)
    fprintf('error\n');
end

fprintf('root mean square %f\n', gof.rmse);

saveas(gcf, 'gof.rmse.svg');

% pause(0.01);


