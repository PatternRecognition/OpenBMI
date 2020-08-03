function [m] = icserializerrapid_delete(m)
% Copyright (C) 2009-2011  EPFL (Ecole Polytechnique Fédérale de Lausanne)
% Michele Tavella <michele.tavella@epfl.ch>
%
% ICSERIALIZERRAPID_DELETE Deletes an ICSerializerRapid object
%
% M = ICSERIALIZERRAPID_DELETE(M) frees the instance of an ICSerializerRapid object 
% pointed by the handle M. 
%
% Returns 0 upon success, >0 otherwise.
%
% See also ICSERIALIZERRAPID_NEW
	mex_id_ = 'delete(i ICSerializerRapid*)';
tobiic(mex_id_, m);
	m = 0;

