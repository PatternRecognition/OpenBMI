function sequence = findReverseSequence(srchStr, LUT),
% Finds the state steps that are necessary for spelling a given string
% (srchStr).

sequence = [];
for charIdx = 1:length(srchStr),
    for state = 1:length(LUT),
        for dir = 1:length(LUT(state).direction),
            if strcmp(LUT(state).direction(dir).label, upper(srchStr(charIdx))),
                sequence = [sequence state-1 dir];
            end
        end %dir
    end %state
end %charIdx

