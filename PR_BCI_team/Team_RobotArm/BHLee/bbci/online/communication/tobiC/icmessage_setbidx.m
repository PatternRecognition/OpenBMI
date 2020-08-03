function [bidx] = icmessage_setbidx(q, value)
% Copyright (C) 2009-2011  EPFL (Ecole Polytechnique Fédérale de Lausanne)
% Michele Tavella <michele.tavella@epfl.ch>
%
% ICMESSAGE_SETBIDX Sets the block number
%
% [BIDX] = ICMESSAGE_SETBIDX(Q, VALUE) sets the block number according to VALUE
% and returns the block number value just because it's cool.
%
% See also ICMESSAGE_GETBIDX, ICMESSAGE_INCBIDX
				mex_id_ = 'o int = setbidx(i ICMessage*, i int)';
[bidx] = tobiic(mex_id_, q, value);

