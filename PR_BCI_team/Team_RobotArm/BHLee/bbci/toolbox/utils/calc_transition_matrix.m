function mc = calc_transition_matrix(file,classes);
%CALC_TRANSITION_MATRIX CALCULATES THE TRANSITION MATRIX OF A EEG-FILE
% 
% usage: 
%    mc = calc_transition_matrix(file,<classes>);
%
% input:
%    file     the name of an EEG file, absolutely or relatively to EEG_MAT_DIR
%    classes  a set of classes as cell which is to used (default: all available classes)
%
% output:
%    mc       the transition matrix. in rows: the predecessor, in column: the successor, the classes were ordered regarding the order in classes
%
% Guido Dornhege, 03/09/2004


[dum,mrk] = loadProcessedEEG(file,[],'mrk');

if exist('classes','var') & ~isempty(classes)
  mrk = mrk_selectClasses(mrk,classes);
end

mc = separate_predecessor(mrk);


mc = sum(mc.y,2);

mc = reshape(mc,[sqrt(length(mc)),sqrt(length(mc))]);

