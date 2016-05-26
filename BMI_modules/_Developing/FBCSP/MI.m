function [Iw] = MI( data, label)
% input:    data = nTrials x csp_w (137 x (8x9))
%           label = nTrials x 2
    %% step 1: Initialization
    % Initialize set of features
    [ntr nl] = size(label);
    [nt nc] = size(data);
    
   %% Step 2: Compute the mutual information of each feature
    % distribution of data and label 
    Pl = sum(label)/ntr;  % P(label): Pl =    0.5036    0.4964
    Hl = -sum( Pl .* log2( Pl ) );    % entropy of Pl: H(label)
    
    for j = 1:nc
        cond_entropy = 0;
        for i = 1:nt
            Pf = 0;
            for w = 1:nl
                X = data(find(label(:,w)~=0),j);
                d = data(i,j); % the data to estimate the density
                estPf(w) = parzen_de(X, d);
                priorP(w) = estPf(w).*Pl(w);    % p(fji)
            end
            Pf = sum(priorP);
            for ww = 1:nl
                bayesP = priorP(ww)./ Pf;    % the probability p(w|fji)
                entropy(ww) = bayesP .* log2( bayesP );
            end
            cond_entropy = cond_entropy + entropy;
        end
        Hd(j) = -sum(cond_entropy);
        Iw(j) = Hl - Hd(j);
    end
end


function [pd] =parzen_de(X,d)
    %Parzen Density Estiamation using Gaussian Kernel
     % FUNCTION
     %   [px] =parzen_de(X,d,h)
     % % INPUT ARGUMENTS:
     %   X:  Training Data for the class
     %   d:  the data we need to estimate the density for
     %   h:  the kernel width
     % OUTPUT ARGUMENTS:
     %   pd: parzen pdf estimate for d
     [Nm l]=size(X);
     pd=0; 
     %!-------------------------------flexible
     yhap = 0;
     for i=1:Nm
        xii=X(i,:);
        yi(i) = d - xii;
        yhap = yhap + yi(i);
     end
     ymean = yhap./Nm;
     ystd = (sum((yi - ymean).^2)./Nm).^(1/2);
     %!------------------------------------------------
     
     for i=1:Nm
        xi=X(i,:);
        y = d - xi;
        stdv = ystd;
%         stdv = 0.25;
        h = (4/(3*Nm)).^(1/5).* stdv;

        pd=pd+exp((-1/2).*(y^2)./(h^2));
     end
     pd=pd.*((1/Nm)*(1/sqrt(2*pi)));
 end