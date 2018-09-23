function [marker, old_marker] = prep_defineClass(marker, varargin )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PREP_DEFINECLASS - defines numbers of markers to the class names
% prep_defineClass (Pre-processing procedure):
% 
% Synopsis:
%     [MARKER] = prep_defineClass(MARKER,<MARKER>)
%
% Example:
%     [marker,original]=prep_defineClass(marker,{'1','left';'2','right'})
%     [marker]=prep_defineClass(marker,{'1','left';'2','right';'3','foot'})
% 
% Arguments:
%     marker     - Loaded marker information, (See Load_BV_mrk)
%     varargin   - Nx2 size cell. Class names should be paird with their
%                  corresponding trigger numbers.
% Returns:
%     marker     - Renewed marker, in a form of OpenBMI
%     old_marker - Same as input marker
% 
% Description:
%     This function defines numbers of markers to the class names, 
%     deleting undefined classes.
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Min-Ho, Lee
% mhlee@image.korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isempty(varargin)
    error('Parameter is misssing: varargin is empty');
end
if ~isfield(marker,'y') && isfield(marker,'t') && isfield(marker,'class') 
    warining('Parameter is missing: .y .t .class');
end

old_marker=marker; marker=[];
mrk_define=varargin{:};
[n_class, ~]=size(mrk_define);
nc_all=logical(1:length(old_marker.y)); 
for i=1:n_class
    [nc]=find(old_marker.y==str2num(mrk_define{i}));
    for j=1:length(nc)
        marker.y_class{nc(j)}=mrk_define{i,2};
    end
    nc_all(nc)=0;
end
marker.y_dec=old_marker.y(~nc_all);
marker.t=old_marker.t(~nc_all);
marker.y_class=marker.y_class(~nc_all);
marker.nClasses=length(mrk_define);
marker.class=varargin{:};
marker.y_logic= zeros(size(mrk_define,1), numel(marker.y_dec)); %% 마커 클래스가 하나일때 오류
for i=1:n_class
    c_n=str2num(cell2mat(mrk_define(i)));
    [~, idx]=find(marker.y_dec==c_n);
    marker.y_logic(i,idx)=1;    
end
marker.y_logic=logical(marker.y_logic);
end