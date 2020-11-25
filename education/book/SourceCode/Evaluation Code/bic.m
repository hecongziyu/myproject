%
% Bayesian Information Criterion
%
function res = bic(expected, actual, nparameters)
   if( length(actual)~=length(expected) ),
       disp('ERROR! Actual and Predicted need to be equal lengths!!!');
       res = NaN;
   else
       nrows  = numel(actual);
	   res = 2*loglikelihood(actual, expected) + nparameters * log(nrows);
   end;
end
