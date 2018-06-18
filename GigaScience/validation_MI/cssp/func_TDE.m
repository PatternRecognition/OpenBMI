function SMTout=time_delay_embedding(SMT,CNT,tau,time_interval)
% TDE: time-delay embedding

SMT_in=SMT;
CNT_in=CNT;
interval=time_interval;

SMT_in.x=permute(SMT_in.x,[1,3,2]);
SMT_in_d=prep_segmentation(CNT_in, {'interval', interval-tau});
SMT_in_d.x=permute(SMT_in_d.x,[1,3,2]);
SMT_in.x=[SMT_in.x,SMT_in_d.x];
SMT_in.x=permute(SMT_in.x,[1,3,2]);

SMTout=SMT_in;
end