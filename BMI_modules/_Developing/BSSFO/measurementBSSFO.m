% File Name: measurement.m
% Author: Heung-Il Suk
% Cite: H.-I. Suk and S.-W. Lee, "A Novel Bayesian Framework for Discriminative 
% Feature Extraction in Brain-Computer Interfaces," IEEE Trans. on PAMI,
% 2012. (Accepted)

function [updatedBSSFO, x_flt, CSPFilter, features] = measurementBSSFO( X1, X2, fs, oldBSSFO, numCSPPatterns, verbose )
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


fprintf( '\tComputing Mutual Information...\n\t\t' );
% for each filter bank
for i=1:oldBSSFO.numBands
    if verbose
        if mod(i, 5) == 0
            fprintf( '%d', i );
        else
            fprintf( '.' );
        end
        if mod(i, 100) == 0
            fprintf( '\n' );
        end
    end
    % It is free to use your own method to compute mutual information.
    miValue(i) = mutual_information( features{1, i}, features{2, i}, kernelWidth );
end
miValue
if verbose == 1
    fprintf( '\n' );
end

updatedBSSFO = oldBSSFO;
updatedBSSFO.miValue = miValue;
expMIValue = exp(miValue);
updatedBSSFO.weight = expMIValue ./ sum(expMIValue);


