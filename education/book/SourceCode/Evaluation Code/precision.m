% RES = precision(ACTUAL, EXPECTED)
%
% PRECISION
%
% ACTUAL - acctual value
% PREDICTED - predicted value
% RES - Precision, if error occurrs it's set to NaN

function res = precision(actual, expected)
   if( length(actual)~=length(expected) ),
       disp('ERROR! Actual and Predicted need to be equal lengths!!!');
       res = NaN;
	   return;
   end;
   
   threshold = 0.5;
   projected = expected >= threshold;
      
   true_positive = sum( actual.*projected );
   true_predicted = sum( projected );
   
   res = true_positive / true_predicted;
   
end
