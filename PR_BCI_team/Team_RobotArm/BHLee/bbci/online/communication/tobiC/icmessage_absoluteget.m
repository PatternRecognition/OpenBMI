function [stoc] = icmessage_absoluteget(q)
% Copyright (C) 2009-2011  EPFL (Ecole Polytechnique Fédérale de Lausanne)
% Michele Tavella <michele.tavella@epfl.ch>
%
% ICMESSAGE_ABSOLUTEGET Returns the absolute as string
%
% [TOC] = ICMESSAGE_ABSOLUTEGET(Q) returns the internal absolute absolute as a
% string.
%
% See also ICMESSAGE_ABSOLUTETIC, ICMESSAGE_ABSOLUTETOC
						stoc = '';
	mex_id_ = 'absoluteget(i ICMessage*, io cstring[x])';
[stoc] = tobiic(mex_id_, q, stoc, 4096);

