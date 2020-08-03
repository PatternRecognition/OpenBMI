function udp = bbci_bet_feedbacks_ERP_Speller_ErrP_detection(udp, bbci)

persistent markerCue old_timestamp_speller old_timestamp_ErrP last_cfy_speller last_cfy_ErrP counter outputs

cfy_speller = udp(1);
cfy_ErrP = udp(2);
marker_speller = udp(3);
timestamp_speller = udp(4);
marker_ErrP = udp(5);
timestamp_ErrP = udp(6);

if timestamp_speller == bbci.minDataLength || isempty(markerCue),
    % initialize the cue
    markerCue = nan([1,10]);
    counter = zeros(1, 6);
    outputs= zeros(1, 6);
    old_timestamp_speller= -inf;
    last_cfy_speller = -inf;
end
if timestamp_speller == bbci.minDataLength || isempty(markerCue),
    markerCue = nan([1,10]);
    old_timestamp_ErrP= -inf;
    last_cfy_ErrP = -inf;
end

if timestamp_speller>old_timestamp_speller && marker_speller > 0
  counter(marker_speller)= counter(marker_speller)+1;
  % insert into cue and set 
  idx = find(isnan(markerCue), 1);
  markerCue(idx) = marker_speller;
%    fprintf('new marker: %03d at %g', marker, timestamp-old_timestamp);
%    fprintf(' -> marker cue: %s\n', vec2str(markerCue));
%    fprintf('counter: %s\n', vec2str(counter));
  old_timestamp_speller = timestamp_speller;
end
if timestamp_ErrP>old_timestamp_ErrP && marker_ErrP > 0,
  fprintf('\nmarker2: %d\n\n', marker_ErrP)
  idx = find(isnan(markerCue), 1);
  markerCue(idx) = marker_ErrP;
  old_timestamp_ErrP = timestamp_ErrP;
end

if ~isnan(cfy_speller) && timestamp_speller-last_cfy_speller < 100,
  cfy_speller= NaN;
  fprintf('blocked classifier output at %g\n', timestamp_speller-last_cfy_speller);
end
if ~isnan(cfy_ErrP) && timestamp_ErrP-last_cfy_ErrP < 100,
  cfy_ErrP= NaN;
  fprintf('blocked classifier output at %g\n', timestamp_ErrP-last_cfy_ErrP);
end

if isnan(cfy_speller) && isnan(cfy_ErrP),
    markerFromCue = 0;
    classifier = NaN;
else
    % get from cue and shift/delete
    markerFromCue = markerCue(1);
    markerCue= [markerCue(2:end), NaN];
    fprintf('%03d -> %.3f / %.3f at %g/%g\n', markerFromCue, cfy_speller, cfy_ErrP, timestamp_speller-last_cfy_speller, timestamp_ErrP-last_cfy_ErrP);
    fprintf('marker cue: %s\n', vec2str(markerCue));
    if ismember(markerFromCue, 1:6) % send speller classifier
      last_cfy_speller= timestamp_speller;
      outputs(markerFromCue)= outputs(markerFromCue) + cfy_speller;
      if sum(counter)==6*bbci.setup_opts.nr_sequences,
        outputs
        [mm,mi]= min(outputs);
        fprintf('\n*** Selected: %d\n\n', mi)
        counter= zeros(1, 6);
        outputs= zeros(1, 6);
      end
      classifier = cfy_speller;
    elseif ismember(markerFromCue, 7:12) % send ErrP classifier
      last_cfy_ErrP= timestamp_ErrP;
      if cfy_ErrP < 0,
        classifier = int16(1); % error detected
      else
        classifier = int16(0); % no error
      end
    else
      error('unknown marker value!')
    end
end

% early_stopping_flag = 0; % dummy for EarlyStopping-implementations

udp = [classifier; markerFromCue];

