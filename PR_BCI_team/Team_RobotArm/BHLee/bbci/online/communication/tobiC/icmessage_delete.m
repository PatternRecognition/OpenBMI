function [q] = icmessage_delete(q)
% Copyright (C) 2009-2011  EPFL (Ecole Polytechnique Fédérale de Lausanne)
% Michele Tavella <michele.tavella@epfl.ch>
%
% ICMESSAGE_DELETE Deletes an ICMessage object
%
% Q = ICMESSAGE_DELETE(Q) frees the instance of an ICMessage object 
% pointed by the handle Q 
%
% Returns 0 upon success, >0 otherwise.
%
% See also ICMESSAGE_NEW
	mex_id_ = 'delete(i ICMessage*)';
tobiic(mex_id_, q);
	q = 0;

