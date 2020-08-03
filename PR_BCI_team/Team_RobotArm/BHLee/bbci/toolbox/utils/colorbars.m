
function colorbars(taskstr, varargin)
%
% USAGE:  colorbars(taskstr, varargin)
%         colorbars(taskstr, Clim, varargin)
%
% colorbars('equalize')   -   equalize all colorbars and subplots. 
%                             if opt.global_lim is not set, the
%                             minimum and maximum values of the existing
%                             colorbars will be used.
%
% colorbars('delete')     -   delete all colorbars in the figure
%
% colorbars('equdel')     -   first equalize all colorbars and subplots, 
%                             then delete all colorbars
% 
%
% varargin                -   struct or propertylist:
%
%                             .handle: figure handle (default: current figure)
%
%                             NOTE: the following options will be ignored if
%                             taskstr=='delete':
%
%                             .Clim: color limits for the colorbars and the 
%                                    subplots (default: min and max values 
%                                    of the current colorbars)
%                             
%                             .min:  set only the minimum color limit
%
%                             .max:  set only the maximum color limit
%                               
% Simon Scholler, May 2011
%

% use user defined minimum and/or maximum values if given
if nargin>1 && isnumeric(varargin{1})
    opt= propertylist2struct(varargin{2:end});
    opt.min = varargin{1}(1);
    opt.max = varargin{1}(2);    
else
    opt= propertylist2struct(varargin{:});
end


opt= set_defaults(opt, ...
                  'handle', gcf, ...
                  'Clim', [], ...
                  'min', [], ...
                  'max', [], ...
                  'adjustColorbarlabels', 1, ...
                  'nColorbarlabels', 5);
              
% get handles
H = findobj(opt.handle,'type','axes');
H = findobj(H,'flat');
H_sp = H(arrayfun(@(x) isempty(get(x,'Tag')), H));              % subplots
H_cb = H(arrayfun(@(x) strcmpi(get(x,'Tag'),'colorbar'), H));   % colorbars

%%
switch taskstr
    
    case 'equdel'
        colorbars('equalize', opt);
        colorbars('delete');
        
    case 'delete'
        % delete all colorbars in the figure
        arrayfun(@(x) delete(x), H_cb);
        
    case 'equalize'
        % derive min and max colorbar values from colorbars
        if isempty(opt.Clim)
            opt.Clim = [Inf -Inf];
            for n = 1:length(H_cb)
                cl = get(H_cb(n),'YLim');
                if cl(1)<opt.Clim(1)
                    opt.Clim(1) = cl(1);
                end
                if cl(2)>opt.Clim(2)
                    opt.Clim(2) = cl(2);
                end
            end
        end
        
        % use user-defined min/max values if given
        if ~isempty(opt.min)
            opt.Clim(1) = opt.min;
        end
        if ~isempty(opt.max)
            opt.Clim(2) = opt.max;
        end
        
        % apply global colorbar limits to subplots and colorbars
        for n = 1:length(H_sp)
            set(H_sp(n),'CLim',opt.Clim)
        end
        for n = 1:length(H_cb)
            set(H_cb(n),'YLim',opt.Clim)
            cbc = get(H_cb(n),'Children');
            if opt.adjustColorbarlabels
                ticks = linspace(opt.Clim(1), opt.Clim(2), opt.nColorbarlabels);
                set(H_cb(n),'YTick', ticks);
            end            
            for c = 1:length(cbc)
                if strcmpi(get(cbc(c),'Type'),'image')
                    set(cbc(c),'YData',opt.Clim)
                end
            end
        end
        
    otherwise
        error('Input string unknown.')
end

