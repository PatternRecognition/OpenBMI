function udp = bbci_bet_feedbacks_ERP_Speller(udp, bbci, mrk_from_condition);

persistent nIterations nClasses buffer counter

if isempty(buffer),
  nClasses= bbci.setup_opts.nClasses;
  nIterations= bbci.setup_opts.nIterations;
  counter= zeros(1, nClasses);
  buffer= zeros(1, nClasses);
  fprintf('[ERP_Speller:] Initializing for %d iterations.\n', ...
          nIterations);
end

classifier = udp(1);
udp= [NaN; mrk_from_condition];
if isnan(classifier),
  return;
end

fprintf('[ERP_Speller:] Received marker #%03d with output %g\n',...
        mrk_from_condition, classifier);
class_idx= mod(mrk_from_condition, 10);  %% Specific assumption
if class_idx==0 || class_idx>nClasses,
  %% error or warning?
  error('[ERP_Speller:] Marker unrecognized.\n');
  return;
end
counter(class_idx)= counter(class_idx) + 1;
if counter(class_idx)>nIterations,
  error('[ERP_Speller:] Too much of class %d received.\n',...
        class_idx);
end
buffer(counter(class_idx), class_idx)= classifier;
if all(counter>=nIterations),
  score= mean(buffer, 1);
  [max_score, selected_class]= min(score);
  fprintf('[ERP_Speller:] Scores: [ %s ] -> selected class %d.\n',...
          vec2str(score, '%5.1f', ' | '), selected_class);
  udp= [selected_class; mrk_from_condition];
  counter(:)= 0;
  buffer(:)= 0;
end
