function sp_addLegends(varargin)
    % IN:       varargin     -  1 cell array for each legend (same arguments 
    %                           as those normally passed to 'legend') for the 
    %                           last added plots

    global SP
    
    nLegends = length(varargin);
    for leg = 1:nLegends
        if ~isempty(varargin{leg})
            SP.LEG{length(SP.AX)-nLegends+leg} = varargin{leg};
        end
    end