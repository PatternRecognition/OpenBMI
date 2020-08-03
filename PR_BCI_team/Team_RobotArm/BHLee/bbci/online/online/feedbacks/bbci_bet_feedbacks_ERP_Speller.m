function udp = bbci_bet_feedbacks_ERP_Speller(udp, bbci);

persistent markerCue old_timestamp last_cfy counter outputs nClasses nSequences lasttime

classifier = udp(1);
marker = udp(2);
timestamp = udp(3);

thistime= clock;
if isempty(lasttime),
  restart= 1;
else
  time_passed= etime(thistime, lasttime)*1000;
  toe= adminMarker('query', [-time_passed 0]);
  restart= ~isempty(intersect(toe, 240));
end
if restart,
    % initialize the cue
    fprintf('initializing %s.\n', mfilename);
    if isfield(bbci, 'marker_output'),
      nClasses= max(bbci.marker_output.value);
    else
      nClasses= 6;  %% default for the Hex paradigm
    end
    if isfield(bbci.setup_opts, 'nr_sequences'),
      nSequences= bbci.setup_opts.nr_sequences;
    else
      nSequences= 10;
    end
    markerCue = ones([1,10])*NaN;  % buffer markers while classifier output
                                   % is calculated
    counter = zeros(1, nClasses);
    outputs= zeros(1, nClasses);
    old_timestamp= -inf;
    last_cfy= -inf;
end
lasttime= thistime;

if timestamp>old_timestamp,
  if marker > 0,
    counter(marker)= counter(marker)+1;
    % insert into cue and set 
    idx = find(isnan(markerCue));
    markerCue(idx(1)) = marker;
    fprintf('new marker: %03d at %g', marker, timestamp-old_timestamp);
    fprintf(' -> marker cue: %s\n', vec2str(markerCue));
    fprintf('counter: %s\n', vec2str(counter));
    old_timestamp= timestamp;
  end
end

if ~isnan(classifier) && timestamp-last_cfy < 100,
  classifier= NaN;
  fprintf('blocked classifier output at %g\n', timestamp-last_cfy);
end

if isnan(classifier),
    markerFromCue = 0;
else
    % get from cue and shift/delete
    markerFromCue = markerCue(1);
    markerCue= [markerCue(2:end), NaN];
    fprintf('%03d -> %.3f at %g\n', markerFromCue, classifier, timestamp-last_cfy);
    fprintf('marker cue: %s\n', vec2str(markerCue));
    last_cfy= timestamp;
    outputs(markerFromCue)= outputs(markerFromCue) + classifier;
    if sum(counter)>=nClasses*nSequences,
      outputs
      [mm,mi]= min(outputs);
      fprintf('\n*** Selected: %d\n\n', mi)
      counter= zeros(1, nClasses);
      outputs= zeros(1, nClasses);
    end
end

udp = [classifier; markerFromCue];

