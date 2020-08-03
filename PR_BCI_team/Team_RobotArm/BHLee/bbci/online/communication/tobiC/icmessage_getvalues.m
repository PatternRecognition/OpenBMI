function [id, key] = icmessage_getvalues()
% Copyright (C) 2009-2011  EPFL (Ecole Polytechnique Fédérale de Lausanne)
% Michele Tavella <michele.tavella@epfl.ch>
%
% ICMESSAGE_GETVALUES Returns the supported value types
%
% [ID, KEY] = ICMESSAGE_GETVALUES() returns the numerical ids (ID) and the
% associated keys (KEY) for all the supported IC values.
%
% See also ICMESSAGE_GETVALUES
	id  = [-1 0 1 2 3];
	key = {'undef' 'prob' 'dist' 'clbl' 'rcoe'};

