function [bidx] = icmessage_incbidx(q)
% Copyright (C) 2009-2011  EPFL (Ecole Polytechnique Fédérale de Lausanne)
% Michele Tavella <michele.tavella@epfl.ch>
%
% ICMESSAGE_INCBIDX Increments the block number
%
% [BIDX] = ICMESSAGE_INCBIDX(Q) increments the block number and returns its
% value
%
% See also ICMESSAGE_GETBIDX, ICMESSAGE_SETBIDX
				mex_id_ = 'o int = incbidx(i ICMessage*)';
[bidx] = tobiic(mex_id_, q);

