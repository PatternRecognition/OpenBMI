function dualCues = find_dual_cues(numCues, distance)
if numCues < 4,
    error('No dual cue can be found when there are less then 4 locations');
end

dualCues = [;];
for ii = 1:numCues-1
    for jj = ii:numCues
        if (abs(ii-jj)>distance) && (abs(ii-jj)<(numCues-distance))
            dualPos = size((dualCues)+1,2)+1;
            dualCues(1,dualPos) = ii;
            dualCues(2,dualPos) = jj;
        end
    end
end