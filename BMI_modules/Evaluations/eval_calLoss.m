function [ loss loss01 ] = eval_calLoss( True_label, cf_out )
%EVAL_CALLOSS Summary of this function goes here
%   Detailed explanation goes here
%only if binary class
if size(True_label,1)==2 %logical
    temp=find(True_label(2,:)==0);
    True_label(2,temp)=-1;
    True_label=True_label(2,:)';
    loss01=sign(cf_out)~=True_label;
else
    Est_label= 1.5 + 0.5*sign(cf_out)';
    loss01=Est_label~=True_label;
end
loss=length(find(loss01)==1)/length(cf_out);
end

