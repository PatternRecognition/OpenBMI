function [toc] = icmessage_relativetoc(q)
% Copyright (C) 2009-2011  EPFL (Ecole Polytechnique Fédérale de Lausanne)
% Michele Tavella <michele.tavella@epfl.ch>
%
% ICMESSAGE_RELATIVETOC Tocs the relative
%
% [TOC] = ICMESSAGE_RELATIVETOC(Q) tocs the internal relative relative
% absolute, returning the time difference in milliseconds.
%
% See also ICMESSAGE_RELATIVETIC, ICMESSAGE_RELATIVEGET
				mex_id_ = 'o double = relativetoc(i ICMessage*)';
[toc] = tobiic(mex_id_, q);

