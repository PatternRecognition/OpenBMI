function [ out ] = prep_removeClass( dat, class )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prep_removeClass (Pre-processing procedure):
%
% Synopsis:
%     [out] = prep_removeClass(DAT,<OPT>)
%
% Example :
%     out = prep_removeClass(dat,{'right','foot'});
%
% Arguments:
%     dat - Structure. Continuous data or epoched data
%     varargin - struct or property/value list of optional properties:
%          : class - Name of classes that you want to delete (e.g. {'right','left'})
%           
% Returns:
%     out - Data structure which deleted class (continuous or epoched)
%
%
% Description:
%     This function removes specific classes
%     from continuous or epoched data.
%     continuous data should be [time * channels]
%     epoched data should be [time * channels * trials]
%
% See also 'https://github.com/PatternRecognition/OpenBMI'
%
% Min-ho Lee, 12-2017
% minho_lee@korea.ac.kr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% if iscellstr(class)
%     warning('check the class parameter, it should be a string');
% end

if ndims(dat.x)==2 && length(dat.chan)>1
    type='cnt';
elseif ndims(dat.x)==3 || (ndims(dat.x)==2 && length(dat.chan)==1)
    type='smt';
end

if ischar(class)
    rm_Class{1}=class;
else
    rm_Class=class(:);
end

out=dat;
n_c=ones(1,length(dat.y_class));
for i=1:length(rm_Class)
    n_c= n_c .* ~ismember(dat.y_class, rm_Class{i});
end
n_c=logical(n_c);
%     msg=sprintf('OpenBMI: class "%s" is deleted',rm_Class{i});
%     disp(msg);
    
    if strcmp(type, 'smt')
        if isfield(dat, 'x')
            out.x=dat.x(:,n_c,:);
        end
    end    
    
    if isfield(dat, 't')
        out.t=dat.t(n_c);
    end
    if isfield(dat, 'y_dec')
        out.y_dec=dat.y_dec(n_c);
    end
    if isfield(dat, 'y_logic')
        tm=~ismember(dat.class(:,2),rm_Class);
        out.y_logic=dat.y_logic(tm,n_c);
    end
    if isfield(dat, 'y_class')
        out.y_class=dat.y_class(n_c);
    end
    if isfield(dat, 'class')
        tm=~ismember(dat.class(:,2),rm_Class);
        out.class=dat.class(tm,:);
    end


end


