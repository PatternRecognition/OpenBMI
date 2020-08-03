function [cont_proc,feature,cls,post_proc,marker_output]= bbciutil_load_setup(setup)
%BBCIUTIL_LOAD_SETUP - Load BBCI classifier setup and set default fields
%
%[cont_proc,feature,cls,post_proc,marker_output]=
%     bbciutil_load_setup(SETUP)
%
%Arguments:
% SETUP: Name of BBCI classifier setup file
%
%Returns:
% ...: variables loaded from BBCI classifier setup

post_proc= [];
marker_output= [];
load(setup, 'cont_proc', 'feature', 'cls', 'post_proc','marker_output');

cls = set_defaults(cls, 'condition', [], 'conditionParam', [], 'fv', ...
                          [], 'applyFcn', [], 'C', [], 'integrate', [], ...
                          'bias', [], 'scale', [], 'dist', [], 'alpha', ...
                          [], 'range', [], 'timeshift', []);
feature = set_defaults(feature, 'cnt', [], 'ilen_apply', [], 'proc', ...
                          [], 'proc_param', []);

if ~isempty(cont_proc), 
  cont_proc = set_defaults(cont_proc, 'clab', [], 'proc', [], 'proc_param', []); 
end
if ~isempty(post_proc),
  post_proc = set_defaults(post_proc, 'proc', [], 'proc_param', []);
end
if ~isempty(marker_output)
  marker_output = set_defaults(marker_output, 'marker', [], 'value', [], ...
                               'no_marker', []);
end
