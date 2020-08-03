function udp = bbci_bet_feedbacks_ERP_Speller(udp, bbci, mrk_from_condition);

classifier = udp(1);

if isnan(classifier),
  udp = [classifier; 0];
else
  udp = [classifier; int16(mod(mrk_from_condition, 10))];
end

