function [epo, numRem] = proc_setArtifactNoclass(epo, channels, amplitude),
    % if channels is empty, default is 'EOGh', 'EOGv'
    % if amplitude is empty, default is 70
    
    detr_epo = proc_detrend(epo);
%     detr_epo = epo;
    if isempty(channels); channels = {'EOGv','EOGh'}; end;
    if isempty(amplitude); amplitude = 70; end;

    low = find(detr_epo.t == 0);
    high = length(detr_epo.t);
    artIdx = [];
    
    for channel = 1:length(channels),
        chId = chanind(detr_epo, channels{channel});
        if ~isempty(chId),
            artIdx = [artIdx find(max(abs(detr_epo.x(low:high, chId, :))) > amplitude)'];
        end        
    end
    
    if ~isempty(artIdx),
        artIdx = unique(artIdx);
        epo.y(:,artIdx) = 0;
    end
    numRem = length(artIdx);
end