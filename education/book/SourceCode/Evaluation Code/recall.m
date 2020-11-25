% RES = recall(ACTUAL, EXPECTED)
%
% RECALL
%
% ACTUAL - acctual value
% PREDICTED - predicted value
% RES - Precision, if error occurrs it's set to NaN

function res = recall(actual, expected)
   if( length(actual)~=length(expected) ),
       disp('ERROR! Actual and Predicted need to be equal lengths!!!');
       res = NaN;
	   return;
   end;
   
   threshold = 0.5;
   projected = expected >= threshold;
      
   true_positive = sum( actual.*projected );
   true_actual = sum( actual );
   
   res = true_positive / true_actual;
   
end
