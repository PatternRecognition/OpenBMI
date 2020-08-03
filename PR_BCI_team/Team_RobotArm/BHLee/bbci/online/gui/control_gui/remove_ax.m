function remove_ax(handle)
% REMOVE_AX ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% delete all handles in the struct handle
%
% usage:
%     remove_ax(handle);
%
% input:
%     handle   - struct with graphic handles
%
% Guido Dornhege
% $Id: remove_ax.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $

% get all fields and delete
if isstruct(handle)
  fi = fieldnames(handle);
  for i = 1:length(fi)
    delete(getfield(handle,fi{i}));
  end
end
