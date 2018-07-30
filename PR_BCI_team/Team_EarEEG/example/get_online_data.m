clear all; clc;

%% ear EEG setting
% instantiate the library
disp('Loading the library...');
lib = lsl_loadlib();

% resolve a stream...
disp('Resolving an EEG stream...');
result = {};
while isempty(result)
    result = lsl_resolve_byprop(lib,'type','signal');
end

% create a new inlet
disp('Opening an inlet...');
inlet = lsl_inlet(result{1});

% figure
disp('Now receiving chunked data...');

%% cap EEG setting
global state ;
bbci_acquire_bv('close');
params = struct;
state = bbci_acquire_bv('init', params);
