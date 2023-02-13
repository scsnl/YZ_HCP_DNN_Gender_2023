function[Val,resid] = regress_fast(x,y,Pval)

    % INPUTS
    % x:    nxp regressor matrix of n observations (rows) across
    %       p regressors (columns)
    % y:    Comun vector of responses
    % Pval: Optional, computes P-values of the estimates

    % OUTPUT
    % Val:  Column vector of the estimated coefficients (Pval == false/zero/missing), or
    %       nx2 matrix of the estimated coefficients (first columsn) with p-values (second
    %       column)

    % USE
    % Val = Regression_fast(x,y)

    if nargin == 2
        Pval = false;
    end

    B   = (x'*x)\x'*y;
    Val = B;
    y_fit = x*B;
    resid=y-y_fit;
    if Pval
        y_fit = x*B;
        df  = -diff(size(x));

        s   = (sum((y-y_fit).^2)/df)^0.5;
        se  = (diag(inv(x'*x))*s^2).^0.5;
        T   = B./se;
        P   = (T>=0).*(1 - tcdf(T,df))*2 + (T<0).*(tcdf(T,df))*2;

        Val = [B,P];
    end

    end