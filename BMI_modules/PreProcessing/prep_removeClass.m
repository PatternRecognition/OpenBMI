function [ mrk ] = proc_remove_class( mrk_orig, varargin )
%PROC_REMOVE_CLASS Summary of this function goes here
%   Detailed explanation goes here

if ~iscellstr(varargin) && ~isnumeric(varargin)
    warning('check the class parameter, it shold be a string or number');
end

switch varargin
    case iscellstr(varargin)
        n_c=~ismember(mrk_orig.class,varargin);
        mrk.y=mrk_orig.y(n_c);
        mrk.t=mrk_orig.t(n_c);
        mrk.class={mrk_orig.class{n_c}};
    case isnumeric(varargin)
        n_c=~ismember(mrk_orig.y,4);
        mrk.y=mrk_orig.y(n_c);
        mrk.t=mrk_orig.t(n_c);
        mrk.class={mrk_orig.class{n_c}};
end

end

