function [fitresult, gof] = dispersion_fit(model, x, y, out)
%CREATEFIT(X,Y,YO)
%  Create a fit.
%
%  Data for 'test_fit_y' fit:
%      X Input : x
%      Y Input : y
%      Z Output: out
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.
%
%  See also FIT, CFIT, SFIT.

%  Auto-generated by MATLAB on 22-Oct-2022 15:08:23


%% Fit: 'test_fit_y'.
[xData, yData, zData] = prepareSurfaceData(x, y, out );

% Set up fittype and options.
ft = fittype( model );

% Fit model to data.
[fitresult, gof] = fit( [xData, yData], zData, ft );

% Plot fit with data.
% figure( 'Name', 'test_fit_y' );
% h = plot( fitresult, [xData, yData], zData );
% legend( h, 'test_fit', 'out vs. x, y', 'Location', 'NorthEast', 'Interpreter', 'none' );
% % Label axes
% xlabel( 'x', 'Interpreter', 'none' );
% ylabel( 'y', 'Interpreter', 'none' );
% zlabel( 'out', 'Interpreter', 'none' );
% % grid on
% view( -16.5, -12.4 );


end
