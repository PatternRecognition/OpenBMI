%BBCI_BET_FINISH SAVES THE SZENARIO
%
% Generate all variables that are required in bbci_bet_apply:
% cont_proc, feature, cls
% 
% input:
%   bbci: struct, general setup variable from bbci_bet_prepare
%
% output:
%     cont_proc  see bbci_bet_apply and bbci_alf_analyze
%     cls        see bbci_bet_apply and bbci_alf_analyze
%     feature    see bbci_bet_apply and bbci_alf_analyze
%     post_proc  see bbci_bet_apply and bbci_alf_analyze
%     marker_output  see bbci_bet_apply and bbci_alf_analyze
% Guido Dornhege, 07/12/2004
% $Id: bbci_bet_finish.m,v 1.1 2006/04/27 14:24:38 neuro_cvs Exp $


string = sprintf('bbci_bet_finish_%s',bbci.setup);
% Before calling the finish routines, extract its setup from bbci, and
% all the analyze results (this stores things like CSPatterns)
opt = bbci.setup_opts;
analyze = bbci.analyze;
% There might be remains from these variables from previous setups, so
% clear them all before calling the setup finish routine
cls = [];
feature = [];
cont_proc = [];
post_proc = [];
marker_output = [];
eval(string);

if isfield(bbci.setup_opts, 'func_adaptBias') && ~isempty(bbci.setup_opts.func_adaptBias)
  eval(bbci.setup_opts.func_adaptBias)
end

% Do we need the following anymore???
% $$$   cls = set_defaults(cls, 'condition', [], 'conditionParam', [], 'fv', ...
% $$$                           [], 'applyFcn', [], 'C', [], 'integrate', [], ...
% $$$                           'bias', [], 'scale', [], 'dist', [], 'alpha', ...
% $$$                           [], 'range', [], 'timeshift', []);
% $$$   feature = set_defaults(feature, 'cnt', [], 'ilen_apply', [], 'proc', ...
% $$$                                   [], 'proc_param', []);
% $$$   cont_proc = set_defaults(post_proc, 'clab', [], 'proc', [], 'proc_param', []);
% $$$   post_proc = set_defaults(post_proc, 'proc', [], 'proc_param', []);
% $$$   marker_output = set_defaults(marker_output, 'marker', [], 'value', [], ...
% $$$                                              'no_marker', []);

save_setup;
