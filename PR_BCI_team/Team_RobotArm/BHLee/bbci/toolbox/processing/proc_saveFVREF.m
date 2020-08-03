function fv= proc_saveFVREF(fv);
% Save a vector FVREF globally for later use in bbci_bet_apply.m by the "main" classifier.
% FVREF is typically calculated by a second "classifier" that runs in parallel to the 
% main classifier in order to preserve basline correction values from the start of a trial. 
% Subsequently the main classifier can use FVREF to perform baseline corrections on a sliding data window via
% proc_subtractFVREF.m
%
% (for an example see bbci_bet_finish_LRP.m)
% 
% see also: proc_subtractFVREF.m
% 
% 01/2011 by Benjamin, Michael, David

global FVREF
FVREF= fv.x;
