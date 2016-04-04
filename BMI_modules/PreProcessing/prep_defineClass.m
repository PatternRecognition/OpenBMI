function [ marker, old_marker] = prep_defineClass(marker, varargin )
%MRK_REDIFINE_CLASS Summary of this function goes here
%   Detailed explanation goes here
% [EEG.marker, EEG.markerOrigin]=prep_defineClass(EEG.marker,{'1','left';'2','right';'3','foot'}); 
% [EEG.marker ]=prep_defineClass(EEG.marker,{'1','left';'2','right';'3','foot'}); 
% mrk_define=opt_proplistToCell(mrk_define{:});
% delete undefined classes
if ~isfield(marker,'y') && isfield(marker,'t') && isfield(marker,'class') 
    warining('Parameter is missing: .y .t .class');
end

if isempty(varargin)
    error('Parameter is misssing: varargin is empty');
end
old_marker=marker; marker=[];
mrk_define=varargin{:};
[nclass temp]=size(mrk_define);
nc_all=logical(1:length(old_marker.y));
for i=1:nclass
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
%logical Y lable
marker.y_logic= zeros(length(mrk_define), numel(marker.y_dec));
for i=1:nclass
    c_n=str2num(cell2mat(mrk_define(i)));
    [temp idx]=find(marker.y_dec==c_n);
    marker.y_logic(i,idx)=1;    
end
marker.y_logic=logical(marker.y_logic);

end



