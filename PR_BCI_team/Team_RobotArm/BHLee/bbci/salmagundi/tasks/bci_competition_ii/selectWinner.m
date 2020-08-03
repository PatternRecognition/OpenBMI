function winner= selectWinner(out, nChoices, nVotes, policy)

if ~isstruct(policy),
  policy= struct('method', policy);
end

nTrials= length(out)/(nChoices*nVotes);
oo= reshape(out, [nChoices, nVotes, nTrials]);
winner= zeros(1, nTrials);

for it= 1:nTrials,
  o= oo(:,:,it);
  switch(policy.method),
   case 'vote',
    [d,si]= sort(o);
    points= zeros(1, nChoices);
    bonus= [5 3 1];
    for ir= 1:min(nVotes,length(bonus));
      for ii= si(ir,:),
        points(ii)= points(ii) + bonus(ir);
      end
    end
    [d,win]= max(points);
   case 'min',
    [d,win]= min( min(o,[],2) );
   case 'mean',
    [d,win]= min( mean(o,2) );
   case 'median',
    [d,win]= min( median(o,2) );
   case 'ival_mean',
    ival= policy.param;
    [d, si]= sort(o, 2);
    osel= zeros(nChoices, length(ival));
    for ic= 1:nChoices,
      osel(ic,:)= o(ic, si(ic,ival));
    end
    [d,win]= min( mean(osel,2) );
   case 'selected_mean',
    if policy.param<1,
      alpha= policy.param;
      nSel= nVotes-floor(alpha*nVotes);
    else
      nSel= nVotes-policy.param;
    end
    om= mean(o,2);
    diff= abs(o - repmat(om, 1, nVotes));
    [so, si]= sort(diff, 2);
    osel= zeros(nChoices, nSel);
    for ic= 1:nChoices,
      osel(ic,:)= o(ic, si(ic,1:nSel));
    end
    [d,win]= min( mean(osel,2) );
   case 'trimmed_mean',
    alpha= policy.param;
    om= mean(o, 2);
    os= std(o, 0, 2);
    osel= o;
    for ic= 1:nChoices,
      lower_lim= om(ic) - alpha*os(ic);
      idx= find(o(ic,:)<lower_lim);
      osel(ic,idx)= lower_lim;
      upper_lim= om(ic) + alpha*os(ic);
      idx= find(o(ic,:)>upper_lim);
      osel(ic,idx)= upper_lim;
    end
    [d,win]= min( mean(osel,2) );  
   otherwise
    error('policy not known');
  end
  winner(it)= win;
end
