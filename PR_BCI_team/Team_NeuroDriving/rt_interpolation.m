function [interRT]= rt_interpolation(cnt,mrk)

    % Calculation of response time between start Devi and end Devi
    clear rt rtPos;
    for i = 1 : length(mrk.pos) / 2
        rt(i) = mrk.pos(i * 2) - mrk.pos(i * 2 - 1);
        rtPos(i) = mrk.pos(i * 2);
    end
    % Interpolation of response times
    interRT = zeros(length(cnt.x), 1);
    % Decision of deviation duration based on averaged values in specific
    % range (First half and last half samples for start and end point)
    interRT(1) = mean(rt(1 : round(length(rt) / 2)));
    interRT(end) = mean(rt(round(length(rt) / 2) : end));
    for i = 1 : length(rt)
        if i == length(rt)
            interRT(1 : rtPos(1)) = linspace(interRT(1), rt(1), length(1 : rtPos(1)));
            interRT(rtPos(i) : end) = linspace(rt(i), interRT(end), length(rtPos(i) : length(interRT)));
        else
            interRT(rtPos(i) : rtPos(i + 1)) = linspace(rt(i), rt(i + 1), length(rtPos(i) : rtPos(i + 1)));
        end
    end