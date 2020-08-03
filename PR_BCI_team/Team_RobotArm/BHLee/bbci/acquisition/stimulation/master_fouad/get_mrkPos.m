function mrk_pos = get_mrkPos(mrk,mrk_num)

mrko = mrk.desc;

% if length(mrk_num.Stim) > 1
    
    for iii = 1:length(mrk_num.Stim)
      
      if mrk_num.Stim(iii)<10
         mrko_i = regexp(mrko, ['S  ',int2str(mrk_num.Stim(iii))]);
      else
        mrko_i = regexp(mrko, ['S ',int2str(mrk_num.Stim(iii))]);
      end
        mrko_ii = cellfun('isempty',mrko_i);


        ii = find(mrko_ii==0);
        ind(iii,:) = ii;
        mrk_pos.Stim(iii,:) = mrk.pos(ind(iii,:));
    end

        mrko_i = regexp(mrko, ['S',int2str(mrk_num.S102)]);

        mrko_ii = cellfun('isempty',mrko_i);


        ii = find(mrko_ii==0);
        ind = ii;
        mrk_pos.S102 = mrk.pos(ind);

        mrko_103 = regexp(mrko, ['S',int2str(mrk_num.S103)]);

        mrko_103i = cellfun('isempty',mrko_103);


        S103 = find(mrko_103i==0);
        
        mrk_pos.S103 = mrk.pos(S103);