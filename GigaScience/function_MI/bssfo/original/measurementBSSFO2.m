% File Name: measurement.m
% Author: Heung-Il Suk
% Cite: H.-I. Suk and S.-W. Lee, "A Novel Bayesian Framework for Discriminative 
% Feature Extraction in Brain-Computer Interfaces," IEEE Trans. on PAMI,
% 2012. (Accepted)

function [updatedBSSFO, x_flt, CSPFilter, features] = measurementBSSFO2( X1, X2, fs, oldBSSFO, numCSPPatterns, verbose )
% function BSSFO = measurement( X, oldBSSFO )
%   input : X1, X2 - raw EEG signal ([channels, sample_points, trials])
%           fs - sampling rate
%           numCSPPatterns - half # of spatial patterns

%% Step 1: Spectral Filtering
% It is free to use your own spectral filtering method.
x_flt = spectral_filtering( X1, X2, fs, oldBSSFO, verbose );

%% Step 2: Spatial Filtering
% It is free to use your own spatial filtering method.
CSPFilter = spatial_filtering( x_flt, oldBSSFO, numCSPPatterns, verbose );

%% Step 3: Feature Extraction
features = feature_extraction( x_flt, CSPFilter, oldBSSFO, verbose );

%% Step 4: Computer Mutual Information
kernelWidth = 1;
miValue = zeros( 1, oldBSSFO.numBands );

% fprintf( '\tComputing Mutual Information...\n\t\t' );
% for each filter bank
for i=1:oldBSSFO.numBands
    if verbose
        if mod(i, 5) == 0
%             fprintf( '%d', i );
        else
%             fprintf( '.' );
        end
        if mod(i, 100) == 0
%             fprintf( '\n' );
        end
    end
    % It is free to use your own method to compute mutual information.
    miValue(i) = mutual_information( features{1, i}, features{2, i}, kernelWidth );

end
% 
% for ii=1:oldBSSFO.numBands
%     %%
%     % input:    data = nTrials x csp_w (137 x (8x9))
% %           label = nTrials x 2
%     %% step 1: Initialization
%     % Initialize set of features
%     data=[features{1, ii} features{2, ii}]' ;
%     label=zeros(100,2);
%     label(1:50,1)=1;  label(51:end,2)=1;
%     [ntr nl] = size(label);
%     [nt nc] = size(data);
%     
%    %% Step 2: Compute the mutual information of each feature
%     % distribution of data and label 
%     Pl = sum(label)/ntr;  % P(label): Pl =    0.5036    0.4964
%     Hl = -sum( Pl .* log2( Pl ) );    % entropy of Pl: H(label)
%     
%     for j = 1:nc
%         cond_entropy = 0;
%         for i = 1:nt
%             Pf = 0;
%             for w = 1:nl
%                 X = data(find(label(:,w)~=0),j);
%                 d = data(i,j); % the data to estimate the density
%                 estPf(w) = parzen_de(X, d);
%                 priorP(w) = estPf(w).*Pl(w);    % p(fji)
%             end
%             Pf = sum(priorP);
%             for ww = 1:nl
%                 bayesP = priorP(ww)./ Pf;    % the probability p(w|fji)
%                 entropy(ww) = bayesP .* log2( bayesP );
%             end
%             cond_entropy = cond_entropy + entropy;
%         end
%         Hd(j) = -sum(cond_entropy);
%         Iw(j) = Hl - Hd(j);
%     end
%     miValue2(ii)=mean(Iw);
% end

if verbose == 1
%     fprintf( '\n' );
end

updatedBSSFO = oldBSSFO;
updatedBSSFO.miValue = miValue;
expMIValue = exp(miValue);
updatedBSSFO.weight = expMIValue ./ sum(expMIValue);


