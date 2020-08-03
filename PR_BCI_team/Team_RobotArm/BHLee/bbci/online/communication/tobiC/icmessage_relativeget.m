function [stoc] = icmessage_relativeget(q)
% Copyright (C) 2009-2011  EPFL (Ecole Polytechnique Fédérale de Lausanne)
% Michele Tavella <michele.tavella@epfl.ch>
%
% ICMESSAGE_RELATIVEGET Returns the relative as string
%
% [TOC] = ICMESSAGE_RELATIVEGET(Q) returns the internal relative relative as a
% string.
%
% See also ICMESSAGE_RELATIVETIC, ICMESSAGE_RELATIVETOC
						stoc = '';
	mex_id_ = 'relativeget(i ICMessage*, io cstring[x])';
[stoc] = tobiic(mex_id_, q, stoc, 4096);
