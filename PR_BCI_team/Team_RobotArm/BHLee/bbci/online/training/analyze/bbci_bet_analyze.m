% bbci_bet_analyze analyzes the data provided by
% bbci_bet_prepare and the specific subject file by means of plots
% and classification performances. Subroutines depending on the setup (as
% given by bbci.setup) are called to do the actual analysis.
%
% input:
%     Cnt    the data
%     mrk    the marker
%     mnt    the montage
%     bbci   general setup variable from bbci_bet_prepare
%     
%     further variables are required by the called function (see
%     documentation there)
%
% output:
%   This routine is a wrapper for analyze routines for individual
%   setups. Thus, the output depends on the setup routines. Check the
%   routines (eg. bbci_bet_analyze_csp)
%            
%
% Guido Dornhege, 07/12/2004
% $Id: bbci_bet_analyze.m,v 1.1 2006/04/27 14:24:38 neuro_cvs Exp $

% Only analyse selected setups:
string = sprintf('bbci_bet_analyze_%s',bbci.setup);
analyze = [];
% Before calling the analyze routine, extract its setup from bbci
global bbci_bet_memo_opt
opt = bbci.setup_opts;
eval(string);
bbci.setup_opts = opt;
bbci_bet_memo_opt = opt;
% Each analyze routine is allowed to generate one result variable to be
% passed on to further routines. The name of this variabe must be
% 'analyze'
bbci.analyze = analyze;

