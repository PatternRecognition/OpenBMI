function fv= proc_subtractFVREF(fv)
% Subtract global (reference) vector from current time frame of EEG data (e.g. during bbci_bet_apply.m)
% FVREF can be used e.g. for maintaining the last baseline correction values that were calculated by a
% second "classifier" that runs in parallel to the main classifier, but only changes values of FVREF
% when the basline interval (e.g. just before the start trial marker) was contained in the current time frame.
%
% (for an example see bbci_bet_finish_LRP.m)
% 
% see also: proc_saveFVREF.m
%
% 01/2011 by Benjamin, Michael, David

global FVREF

if ~isempty(FVREF),
  [T, nC, nE]= size(fv.x);
  fv.x= fv.x - repmat(FVREF, [T 1 nE]);
end
