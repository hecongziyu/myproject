% RES = LOGLIKELIHOOD(ACTUAL, EXPECTED)
%
% Log Likelihood
%
% ACTUAL - acctual value
% PREDICTED - predicted value
% RES - Root Mean Squared Error, if error occurrs it's set to -1

function res = loglikelihood(actual, expected)
   if( length(actual)~=length(expected) ),
       disp('ERROR! Actual and Predicted need to be equal lengths!!!');
       res = NaN;
   else
	   expected(expected==1) = expected(expected==1) - 0.000001;
	   expected(expected==0) =                         0.000001;
	   res = -sum( actual.*log(expected) + (1-actual).*log(1-expected) );
% 	   ACTUAL = double(actual);
% 	   ACTUAL(ACTUAL==0) = -1;
% 	   res =  -sum( log( sigmoid(ACTUAL.*expected) ) );% + C*0.5*w'*w; % negative
   end;
end

