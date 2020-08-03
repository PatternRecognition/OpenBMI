function udp = bbci_bet_feedbacks_1d_AEP(udp,bbci);

persistent markerCue markerNumber clsNumber lasttime
classifier = udp(1);
marker = udp(2);
timestamp = udp(3);


thistime= clock;
if isempty(lasttime),
  lasttime = thistime;
  time_passed = Inf;
else
  time_passed= etime(thistime, lasttime)*1000;
  marker= adminMarker('query', [-time_passed 0]);
end

if timestamp == bbci.minDataLength || isempty(markerCue) || time_passed > 2000,
    % initialize the cue
    disp('Marker cue reset.');
    lasttime = clock;
    markerNumber = 0;
    clsNumber = 0;
    markerCue = ones([1,10])*NaN;
end

if ~isempty(marker) & marker > 0,
    % insert into cue and set 
    for i = 1:length(marker),
        lasttime = clock;
        idx = find(isnan(markerCue));
        if idx(1) == 1 | markerCue(idx(1)-1) ~= marker(i),
            markerCue(idx(1)) = marker(i);
            markerNumber = markerNumber + 1;
        end
    end
end

markerFromCue = 0;
if ~isnan(classifier),
    % get from cue and shift/delete
    lasttime = clock;
    markerFromCue = markerCue(1);
    markerCue(1) = NaN;
    markerCue = circshift(markerCue, [1 -1]);
    clsNumber = clsNumber + 1;
%     if markerFromCue == 20,
%         classifier = -1;
%     else
%         classifier = 1;
%     end
end

udp = [classifier; markerFromCue];

end
