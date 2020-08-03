function [bidx] = icmessage_getbidx(q)
% Copyright (C) 2009-2011  EPFL (Ecole Polytechnique Fédérale de Lausanne)
% Michele Tavella <michele.tavella@epfl.ch>
%
% ICMESSAGE_GETBIDX Gets the block number
%
% [BIDX] = ICMESSAGE_GETBIDX(Q) returns the block number
%
% See also ICMESSAGE_SETBIDX, ICMESSAGE_INCBIDX
				mex_id_ = 'o int = getbidx(i ICMessage*)';
[bidx] = tobiic(mex_id_, q);

