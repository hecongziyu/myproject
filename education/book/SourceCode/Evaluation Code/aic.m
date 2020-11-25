%
% Akaike Information Criterion
%
function res = aic(expected, actual, nparameters)
   if( length(actual)~=length(expected) ),
       disp('ERROR! Actual and Predicted need to be equal lengths!!!');
       res = NaN;
   else
	   res = 2*loglikelihood(actual, expected) + 2 * nparameters;
   end;
end
