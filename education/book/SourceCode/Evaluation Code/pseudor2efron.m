% RES = pseudor2efron(ACTUAL, EXPECTED)
%
% Efron's Pseudo R-squared 
%
% ACTUAL - acctual value
% PREDICTED - predicted value
% RES - Efron's Pseudo R-squared, if error occurrs it's set to NaN
% source: http://statistics.ats.ucla.edu/stat/mult_pkg/faq/general/Psuedo_RSquareds.htm
function res = pseudor2efron(actual, expected)

    if( length(actual)~=length(expected) ),
       disp('ERROR! Actual and Predicted need to be equal lengths!!!');
       res = NaN;
       return;
    end;
    
    res = 1 - sum( (actual-expected).^2 ) / sum( (actual-mean(actual)).^2 );
    
end
