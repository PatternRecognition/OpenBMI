function [smessage] = icmessage_serialize(s)
% Copyright (C) 2009-2011  EPFL (Ecole Polytechnique Fédérale de Lausanne)
% Michele Tavella <michele.tavella@epfl.ch>
%
% ICMESSAGE_SERIALIZE Serializes an ICMessage using an ICSerializer
%
% SMESSAGE = ICMESSAGE_SERIALIZE(S) Serializes the ICMessage Q (set at 
% creation time) using the ICSerializer S.
% Returns an empty string on error.
%
% See also ICMESSAGE_NEW, ICSERIALIZERRAPID_NEW, ICMESSAGE_DESERIALIZE
										smessage = '';
	mex_id_ = 'serialize(i ICSerializerRapid*, io cstring[x])';
[smessage] = tobiic(mex_id_, s, smessage, 4096);

