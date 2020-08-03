function [q] = icmessage_new();
% Copyright (C) 2009-2011  EPFL (Ecole Polytechnique Fédérale de Lausanne)
% Michele Tavella <michele.tavella@epfl.ch>
%
% ICMESSAGE_NEW Allocates an ICMessage object
%
% Q = ICMESSAGE_NEW() returns the handle Q to a new ICMessage object.
%
% See also ICMESSAGE_DELETE
	mex_id_ = 'o ICMessage* = new()';
[q] = tobiic(mex_id_);

