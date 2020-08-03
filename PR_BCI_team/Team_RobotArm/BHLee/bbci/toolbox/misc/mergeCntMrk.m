function cntmrk= mergeCntMrk(cnt, mrk)
%cntmrk= mergeCntMrk(cnt, mrk)
%
% Essentially calls merge_structs. Additionally a field 'continous'
% is created, that tells XVALIDATION that field 'x' does not hold
% epochs but continuous EEG data.

cntmrk= merge_structs(cnt, mrk);
cntmrk.continuous= 1;
