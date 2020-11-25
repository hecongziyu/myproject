% RES = CBD(ACTUAL, EXPECTED)
%
% Capped Binomial Deviance for binary classification/prediction
%
% ACTUAL - acctual value
% PREDICTED - predicted value
% RES - resulting Capped Binomial Deviance, if error occurrs it's set to -1
%
% Sources:
% 1. http://www.kaggle.com/c/ChessRatings2/details/Evaluation
% 2. http://www.kaggle.com/c/PhotoQualityPrediction/forums/t/1013/r-function-for-binomial-deviance
% 	CappedBinomialDeviance <- function(a, p) {
% 		if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
% 		p_capped <- pmin(0.99, p)
% 		p_capped <- pmax(0.01, p_capped)
% 		-sum(a * log(p_capped, base=10) + (1 - a) * log(1 - p_capped, base=10)) / length(a)
% 	}

function res = cbd(actual, expected)
   if( length(actual)~=length(expected) ),
       disp('ERROR! Actual and Predicted need to be equal lengths!!!');
       res = NaN;
   else
       expected_capped = min(0.99, expected);
       expected_capped = max(0.001, expected_capped);
       res = -sum( actual.*log10(expected_capped) + ...
		   (1-actual).*log10(1-expected_capped) ) / length(actual);
   end;
end

