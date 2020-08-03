% validation - cross-validation and model selection
%
% cross-validation
%   doXvalidation       - do cross-validation of a classifier on some features
%   doXvalidationPlus   - as above but with many special features,
%                         especially the classifier can have free model
%                         parameters ("classification model") which are
%                         chosen (by selectModel) on each training set of
%                         the cross-validation procedure
%   calcConfusionMatrix - confusion of classes (like false positives ...)
%                         based on the result of a cross-validation
%   sampleDivisions     - is used the select train/test splits
%
% model selection
%   selectModel         - select model parameters for a classification model
%   selectModelCONDOR   - distributes subcalculations to condor
%
% other utilities
%   trainClassifier     - train a classifier on (some) features
%   applyClassifier     - apply a previously trained classifier to features
%
% other functions in this directory are probably obsolete
%
% note that for evaluating a classification method with model parameters
% selecting the model parameters first and then do cross-validation is a
% bit of cheating, because in the model selection procedure you already
% use information about future test sets. see the help of doXvalidationPlus.
