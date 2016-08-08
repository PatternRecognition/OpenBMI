function [ h, bandPowers ] = plot_onlineBand( h, dat, mrk, freq, bandPowers, plotChannels, state, arteIdx )
%PLOT_ONLINEBAND 이 함수의 요약 설명 위치
%   자세한 설명 위치

if isempty(arteIdx)
    plotIdx = [];
    for i = 1 : length(plotChannels)
        for j = 1 : length(state.clab)
            if strcmp(plotChannels{i}, state.clab{j})
                plotIdx = [plotIdx j];
                break;
            end
        end
    end
    
    
    filt_x = prep_filter(dat(:, plotIdx), {'frequency', freq(1, :); 'fs', state.fs});
    filt_y = prep_filter(dat(:, plotIdx), {'frequency', freq(2, :); 'fs', state.fs});
    
    plot_x = log(var(filt_x));
    plot_y = log(var(filt_y));
    
    bandPowers = [bandPowers; [plot_x plot_y]];
end

for i = 1 : length(plotChannels)
    width = 2;
    
    if isempty(arteIdx)
        switch mrk
            case 1
                pMrk = 'mo';
            case 2
                pMrk = 'c*';
            case 3
                pMrk = 'yx';
            case 4
                pMrk = 'kd';
        end
        plot(h(i), plot_x(i), plot_y(i), pMrk, 'LineWidth', width);
    else
        for j = 1 : length(arteIdx)
            switch mrk(arteIdx(j))
                case 1
                    pMrk = 'ro';
                case 2
                    pMrk = 'r*';
                case 3
                    pMrk = 'rx';
                case 4
                    pMrk = 'rd';
            end
            plot(h(i), bandPowers(j, i), bandPowers(j, i + size(bandPowers, 2) / 2), pMrk, 'LineWidth', width);
        end
    end
    drawnow;
end

