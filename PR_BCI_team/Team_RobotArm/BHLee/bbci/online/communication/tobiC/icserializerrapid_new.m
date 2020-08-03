function [q] = icserializerrapid_new(m);
% Copyright (C) 2009-2011  EPFL (Ecole Polytechnique Fédérale de Lausanne)
% Michele Tavella <michele.tavella@epfl.ch>
%
% ICSERIALIZERRAPID_NEW Allocates an ICSerializerRapid object
%
% M = ICSERIALIZERRAPID_NEW(Q) return the handle M to a new ICSerializerRapid
% object. Q is the handle to an instance of ICMessage.
%
% See also ICSERIALIZERRAPID_DELETE, ICMESSAGE_NEW, ICMESSAGE_DELETE
	mex_id_ = 'o ICSerializerRapid* = new(i ICMessage*)';
[q] = tobiic(mex_id_, m);

