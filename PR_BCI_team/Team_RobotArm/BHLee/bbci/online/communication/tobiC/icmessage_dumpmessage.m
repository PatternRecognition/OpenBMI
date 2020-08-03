function [isempty] = icmessage_dumpmessage(q)
% Copyright (C) 2009-2011  EPFL (Ecole Polytechnique Fédérale de Lausanne)
% Michele Tavella <michele.tavella@epfl.ch>
%
% ICMESSAGE_DUMPMESSAGE Prints an ICMessage
%
% [ISEMPTY] = ICMESSAGE_DUMPMESSAGE(Q) Returns 1 if the ICMessage Q has at least
% one ICClassifier and prints on the standard output the internal structure.
% Returns 0 without any output otherwise.
											mex_id_ = 'o bool = dumpc(i ICMessage*)';
[isempty] = tobiic(mex_id_, q);

