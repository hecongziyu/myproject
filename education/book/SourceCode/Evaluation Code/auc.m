function Error = auc(Predicted,Actual)
        nTarget     = sum(double(Actual == 1));
        nBackground = sum(double(Actual == 0));

        % Rank data
        R = tiedrankk(Predicted);  % 'tiedrank' from Statistics Toolbox

        % Calculate AUC
        Error = (sum(R(Actual == 1)) - (nTarget^2 + nTarget)/2) / (nTarget * nBackground);

