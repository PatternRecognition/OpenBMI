
function sp_addPlots(handles, showColorbar)
    % add plots via this function in order to later visualize it with
    % scrollPlots(), all added figures will subsequently be closed
    % 
    % INPUT: handles        -       a single figure handle or 
    %                               a cell array of figure handles
    %        showColorbar   -       if plot has a colorbar, set this to 1   
    %

    global SP
    
    h = gcf;
    set(h, 'Visible', 'off')
    
    if ~isfield(SP,'AX')    SP.AX = {};       end
    if ~isfield(SP,'CB')    SP.CB = [];       end
    if ~isfield(SP,'LEG')   SP.LEG = {};      end
    if ~isfield(SP,'LE')    SP.LE = lasterr;  end
    if ~exist('showColorbar', 'var') showColorbar = 0; end
    
    if ~strcmpi(SP.LE, lasterr)
        scrollPlots('close')
    end
    if ~isfield(SP,'FIG') || isempty(SP.FIG) || ~ishandle(SP.FIG) 
        SP.FIG = 2000;
        figure(SP.FIG)
        set(SP.FIG, 'Visible', 'off', 'HitTest', 'off', 'NextPlot', 'new')
    end
    if ~exist('handles', 'var') || isempty(handles)
        if isempty(SP.AX)
            figure(h)
            set(h, 'Visible', 'off')
        end
        SP.AX{end+1} = gca;
        SP.LEG{end+1} = [];
        SP.CB(end+1) = showColorbar;
        set(gca, 'Parent', SP.FIG)
        close(h)
    elseif isscalar(handles)
        if iscell(handles)
            handles = handles{1};
        end
        figure(handles)
        SP.AX{end+1} = gca;
        SP.LEG{end+1} = [];
        SP.CB(end+1) = showColorbar;
        set(gca, 'Parent', SP.FIG)
        close(handles)
    else
        for n = 1:length(handles)
            figure(handles{n})
            SP.AX{end+1} = gca;
            SP.LEG{end+1} = [];
            SP.CB(end+1) = showColorbar(n);
            set(gca, 'Parent', SP.FIG)
            close(handles{n})
        end
    end
    SP.LE = lasterr;
    