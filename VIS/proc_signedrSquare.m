function out = proc_signedrSquare(SMT, varargin)
if size(SMT.class, 1) > 2
    return;
elseif size(SMT.class, 1) == 1
    
end

cls1 = SMT.y_logic(1,:);
cls2 = SMT.y_logic(2,:);
ncls1 = length(find(cls1));
ncls2 = length(find(cls2));

div = std(SMT.x, 0, 2);
iConst = find(div==0);
div(iConst) = 1;
rval = ((mean(SMT.x(:,cls1,:), 2) - mean(SMT.x(:,cls2,:),2)) * sqrt(ncls1*ncls2)) ./ (div * (ncls1+ncls2));
rval(iConst) = NaN;

se = 1./(ncls1+ncls2) * ones(size(rval)) -3; % ->> ¾ê´Â ¹»±î..?
SMT.x = rval;
SMT.x = SMT.x .* abs(SMT.x);
SMT.se = se;

SMT.class = {'1', 'sgn r^2'};
SMT = rmfield(SMT,{'t','y_dec','y_logic','y_class'});

out = SMT;
end
% 
% function out = proc_rValue(SMT, varargin)
% % Class 2°³ÀÏ ¶§
% if size(SMT.class, 1) > 2
%     return;
% elseif size(SMT.class, 1) == 1
%     
% end
% [time, trials, channels] = size(SMT.x);
% SMT.x = permute(SMT.x, [1 3 2]);
% SMT.x = reshape(SMT.x, [time*channels, trials]);
% 
% c1 = SMT.y_logic(1,:);
% c2 = SMT.y_logic(2,:);
% lp = length(find(c1));
% lq = length(find(c2));
% 
% div = std(SMT.x, 0, 2);
% iConst = find(div==0);
% div(iConst) = 1;
% rval = ((mean(SMT.x(:,c1), 2) - mean(SMT.x(:,c2),2)) * sqrt(lp*lq)) ./ (div * (lp+lq));
% rval(iConst) = NaN;
% rval = reshape(rval, [time channels 1]);
% 
% [SMT.y_class{:}] = deal('sgnr^2');
% SMT.class = {'1', 'sgnr^2'};
% SMT.y_logic = true(1, length(SMT.y_logic));
% SMT.y_dec = ones(1, length(SMT.y_dec));
% 
% out = SMT;
% out.x = rval;
% end