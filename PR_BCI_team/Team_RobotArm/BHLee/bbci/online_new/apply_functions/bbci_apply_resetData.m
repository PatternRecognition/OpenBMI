function data= bbci_apply_resetData(data)
%BBCI_APPLY_RESETDATA - Reset the data structure of bbci_apply
%
%Synopsis:
%  DATA= bbci_apply_resetData(DATA)
%
%Arguments:
%  DATA - Structure of bbci_apply which holds all current data
%
%Output:
%  DATA - Updated data structure (cleared source, feature, classifer)

% 02-2011 Benjamin Blankertz


%% This is very tidy, but probably not necessary:
[data.source.x]= deal([]);
[data.feature.x]= deal([]);
[data.classifier.x]= deal([]);
[data.control.packet]= deal({});
