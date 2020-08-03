function [status] = icmessage_deserialize(s, smessage)
% Copyright (C) 2009-2011  EPFL (Ecole Polytechnique Fédérale de Lausanne)
% Michele Tavella <michele.tavella@epfl.ch>
%
% ICMESSAGE_DESERIALIZE Deerializes an ICMessage using an ICSerializer
%
% STATUS = ICMESSAGE_DESERIALIZE(S, SMESSAGE) Deserializes SMESSAGE filling up
% ICMessage Q (set at creation time). S is an  ICSerializer.
% Returns 0 on error, 1 upon success.
%
% See also ICMESSAGE_NEW, ICSERIALIZERRAPID_NEW, ICMESSAGE_SERIALIZE
											mex_id_ = 'o bool = deserialize(i ICSerializerRapid*, i cstring[x])';
[status] = tobiic(mex_id_, s, smessage, 4096);

