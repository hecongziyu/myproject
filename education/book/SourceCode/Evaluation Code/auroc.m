% RES = AUROC(ACTUAL, EXPECTED)
%
% Area Under ROC Curve
%
% ACTUAL - acctual value
% PREDICTED - predicted value
% RES - Area Under ROC Curve, if error occurrs it's set to -1

function res = auroc(actual, expected)
   if( length(actual)~=length(expected) ),
       disp('ERROR! Actual and Predicted need to be equal lengths!!!');
       res = NaN;
	   return;
   end;
   
	P = nnz(actual);        % number of non-zeros (1's)
	N = numel(actual) - P;  % number of 0's

	runs = 100;

	tpr = zeros(runs,1); % true positive rate
	fpr = zeros(runs,1); % false positive rate

	for i=1:runs,
		projected = expected>= i/runs;
		tpr(i) = sum(projected.*actual)/P;
		fpr(i) = sum( projected.*(1-actual) )/N;
	end;

	tpr = [1; tpr; 0];
	fpr = [1; fpr; 0];

	delta_xs = fpr(1:(end-1)) - fpr(2:end);
	heights = (tpr(1:(end-1))+tpr(2:end))./2;

	res = sum(heights.*delta_xs);

end
