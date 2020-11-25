% RES = AICC(ACTUAL, EXPECTED, NPARAMETERS)
%
% Akaike Information Criteria with Correction
%
% ACTUAL - acctual value
% PREDICTED - predicted value
% NPARAMETERS - number of parameters in the model
% RES - AIC with correction, if error occurrs it's set to NaN
function res = aicc(actual, expected, nparameters)
    
    % let kernel AIC handle errors
    nrows  = numel(actual);

    res = aic(actual, expected, nparameters) + ...
        2 * nparameters * (nparameters + 1) / (nrows - nparameters  - 1);

end
