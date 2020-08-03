function udp = bbci_bet_feedbacks_ERP_MatrixSpeller(udp, bbci, mrk_from_condition);

persistent nIterations nClasses nCol nRow buffer counter lasttime

thistime= clock;
if isempty(lasttime),
  lasttime= thistime;
end
time_passed= etime(thistime, lasttime)*1000;
toe= adminMarker('query', [-time_passed 0]);
restart= ~isempty(intersect(toe, 250));
if isempty(buffer) | restart,
  nIterations= bbci.setup_opts.nr_sequences;
  nCol= bbci.setup_opts.matrix_columns;
  nRow= bbci.setup_opts.nr_symbols/bbci.setup_opts.matrix_columns;
  nClasses= nCol + nRow;
  counter= zeros(1, nClasses);
  buffer= zeros(1, nClasses);
  fprintf('[MatrixSpeller:] Initializing for %d iterations.\n', ...
          nIterations);
  fprintf('[MatrixSpeller:] Classes : [%s]\n',...
           vec2str(1:nClasses, '%4d', '|'));
end
lasttime= clock;

classifier = udp(1);
udp= [NaN; mrk_from_condition];
if isnan(classifier),
  return;
end

fprintf('[MatrixSpeller:] Received marker #%03d with output %g\n',...
        mrk_from_condition, classifier);
class_idx= mod(mrk_from_condition-10, 20);
if class_idx>nRow,
  class_idx= class_idx - (10-nRow);
end
if class_idx==0 || class_idx>nClasses,
  %% error or warning?
  error('[MatrixSpeller:] Marker unrecognized.\n');
  return;
end
counter(class_idx)= counter(class_idx) + 1;
buffer(counter(class_idx), class_idx)= classifier;
fprintf('[MatrixSpeller:] Counting: [%s]\n',...
          vec2str(counter, '%4d', '|'));

if sum(counter) >= nIterations*nClasses,
  score= mean(buffer, 1);
  [row_score, selected_row]= min(score(1:nRow));
  [col_score, selected_col]= min(score(nRow+[1:nCol]));
  selected_symbol= selected_col + (selected_row-1)*nCol;
  fprintf('[MatrixSpeller:] Scores:  [%s]\n -> selected class %d.\n\n',...
          vec2str(score, '%4.1f', '|'), selected_symbol);
  udp= [int16(selected_symbol-1); mrk_from_condition];
  counter(:)= 0;
  buffer(:)= 0;
end
