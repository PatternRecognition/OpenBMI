function [toc] = icmessage_absolutetoc(q)
% Copyright (C) 2009-2011  EPFL (Ecole Polytechnique Fédérale de Lausanne)
% Michele Tavella <michele.tavella@epfl.ch>%
%
% ICMESSAGE_ABSOLUTETOC Tocs the absolute
%
% [TOC] = ICMESSAGE_ABSOLUTETOC(Q) tocs the internal absolute absolute,
% returning the time difference in milliseconds.
%
% See also ICMESSAGE_ABSOLUTETIC, ICMESSAGE_ABSOLUTEGET
				mex_id_ = 'o double = absolutetoc(i ICMessage*)';
[toc] = tobiic(mex_id_, q);

