% File Name: BSSFO.m
% Author: Heung-Il Suk
% Cite: H.-I. Suk and S.-W. Lee, "A Novel Bayesian Framework for Discriminative 
% Feature Extraction in Brain-Computer Interfaces," IEEE Trans. on PAMI,
% 2012. (Accepted)

function [updatedBSSFO, CSP, LDA, loss] = BSSFO( X1, X2, fs, numCSPPatterns, initBSSFO, niter, kernel, kerneloption, verbose )
%   input : X1, X2 - raw EEG signal ([channels, sample_points, trials])
%           fs - sampling rate
%           numCSPPatterns - half # of spatial patterns
%           initBSSFO - initial particles 
%           niter - # of iterations in posterior pdf estimation
%           kernel, kerneloption - option parameters used in SVM training

%   output : updatedBSSFO
%            CSP - trained spatial patterns, one set for each frequency band
%            xsup, wsvm, w0svm - SVM outputs

for i=1:niter
    fprintf( '%d-th iteration...\n', i );
    
    if i>1
        oldBSSFO = resamplingBSSFO( updatedBSSFO );        
        oldBSSFO = driftBSSFO( oldBSSFO );
    else
        oldBSSFO = initBSSFO;
    end
     
    % x_flt: filtered EEG signals, (# class, # bands)
    % CSP: spatial patterns, (1, # class)
    % features: log(var(transformed EEG signals)), (# class, # bands)
verbose=1;
    [updatedBSSFO, x_flt, CSP, features] = measurementBSSFO( X1, X2, fs, oldBSSFO, numCSPPatterns, verbose );    
end

% Determine filter banks based on the updatedBSSFO sample weights
[val, idx] = sort( updatedBSSFO.weight, 'descend' );
updatedBSSFO.weight = val;
updatedBSSFO.sample = updatedBSSFO.sample(:, idx);
CSP = CSP(idx);
x_flt = x_flt(:, idx);
features = features(:, idx);


%% SVM Training
% We use a "SVM and Kernel Methods Matlab Toolbox" that is available at 
% http://asi.insa-rouen.fr/enseignants/~arakotom/toolbox/index.html
disp( 'Training SVM...' );

lambda = 1e-4;
C = 1e5;
verbose = 1;

kernel='gaussian';
kerneloption=1;
for i=1:updatedBSSFO.numBands
    trainData = [features{1,i} features{2,i}];
    trainLabel = [ones(1, size(features{1, i}, 2)) -ones(1, size(features{2, i}, 2))];
%     [xsup{i}, wsvm{i}, w0svm{i}, pos{i}, tps{i}, alpha{i}] = ...
%         svmclass( trainData', trainLabel', C, lambda, kernel, kerneloption, verbose );
    
    %% LDA cross-validation
    fv.x=trainData;
    fv.y(1,1:length(X1(1,1,:)))=1;fv.y(2,length(X1(1,1,:))+1:length(X1(1,1,:))+length(X2(1,1,:)))=1 
    [loss(i), loss_eeg_std, out_eeg.out, memo] = xvalidation(fv,'LDA');    
    LDA(i)=trainClassifier(fv,'LDA');
end


