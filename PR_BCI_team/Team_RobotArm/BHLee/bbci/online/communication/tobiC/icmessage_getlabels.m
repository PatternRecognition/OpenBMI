function [id, key] = icmessage_getlabels()
% Copyright (C) 2009-2011  EPFL (Ecole Polytechnique Fédérale de Lausanne)
% Michele Tavella <michele.tavella@epfl.ch>
%
% ICMESSAGE_GETLABELS Returns the supported label types
%
% [ID, KEY] = ICMESSAGE_GETLABELS() returns the numerical ids (ID) and the
% associated keys (KEY) for all the supported IC labels.
%
% See also ICMESSAGE_GETLABELS
	id  = [-1 0 1 2];
	key = {'undef' 'biosig' 'class' 'custom'};

