% RES = RMSE(ACTUAL, EXPECTED)
%
% Root Mean Squared Error
%
% ACTUAL - acctual value
% PREDICTED - predicted value
% RES - Root Mean Squared Error, if error occurrs it's set to -1

function res = rmse(actual, expected)
   if( length(actual)~=length(expected) ),
       disp('ERROR! Actual and Predicted need to be equal lengths!!!');
       res = NaN;
   else
	   res = sqrt( mean((actual - expected).^2) );
   end;
end

