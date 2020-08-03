function udp = bbci_bet_feedbacks_RSVP_Speller(udp, bbci, mrk_from_condition);

persistent nIterations nClasses buffer counter lasttime

thistime= clock;
if isempty(lasttime),
  lasttime= thistime;
end
time_passed= etime(thistime, lasttime)*1000;
toe= adminMarker('query', [-time_passed 0]);
restart= ~isempty(intersect(toe, 243));
if isempty(buffer) | restart,
  nClasses= bbci.setup_opts.nClasses;
  nIterations= bbci.setup_opts.nr_sequences;
  counter= zeros(1, nClasses);
  buffer= zeros(1, nClasses);
  fprintf('\n[ERP_Speller:] Initializing for %d iterations.\n', ...
          nIterations);
  fprintf('[ERP_Speller:] Classes : [%s]\n',...
           vec2str(1:nClasses, '%4d', '|'));
end
lasttime= clock;

classifier = udp(1);
udp= [NaN; mrk_from_condition];
if isnan(classifier),
  return;
end

fprintf('[ERP_Speller:] Received marker #%03d with output %g\n',...
        mrk_from_condition, classifier);
class_idx= mod(mrk_from_condition-30, 40);  %% Specific assumption
if class_idx==0 || class_idx>nClasses,
  %% error or warning?
  error('[ERP_Speller:] Marker unrecognized.\n');
  return;
end
counter(class_idx)= counter(class_idx) + 1;
fprintf('[ERP_Speller:] Counting: [%s]\n',...
          vec2str(counter, '%4d', '|'));

%if counter(class_idx)>nIterations,
%  error('[ERP_Speller:] Too much of class %d received.\n',...
%        class_idx);
%end
buffer(counter(class_idx), class_idx)= classifier;
%if all(counter>=nIterations),
if sum(counter) >= nIterations*nClasses,
  score= mean(buffer, 1);
  [max_score, selected_class]= min(score);
  fprintf('[ERP_Speller:] Scores:   [%s]\n  -> selected class %d.\n\n',...
          vec2str(score, '%4.1f', '|'), selected_class);
  udp= [int16(selected_class-1); mrk_from_condition];
  counter(:)= 0;
  buffer(:)= 0;
end
