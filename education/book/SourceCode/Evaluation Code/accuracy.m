% RES = accuracy(ACTUAL, EXPECTED)
%
% ACCURACY
%
% ACTUAL - acctual value
% PREDICTED - predicted value
% RES - Accuracy, if error occurrs it's set to NaN

function res = accuracy(actual, expected)
   if( length(actual)~=length(expected) ),
       disp('ERROR! Actual and Predicted need to be equal lengths!!!');
       res = NaN;
	   return;
   end;
   
   threshold = 0.5;
   projected = expected >= threshold;
      
   res = mean(actual == projected);
   
end
