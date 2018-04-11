function out = proc_rSquareSigned(SMT, varargin)
    SMT = proc_rValue(SMT);
    SMT.x = SMT.x .* abs(SMT.x);
    out = SMT;
end

function out = proc_rValue(SMT, varargin)
% Class 2°³ÀÏ ¶§
if size(SMT.class, 1) > 2
    return;
elseif size(SMT.class, 1) == 1

end
[time, trials, channels] = size(SMT.x);
SMT.x = permute(SMT.x, [1 3 2]);
SMT.x = reshape(SMT.x, [time*channels, trials]);

c1 = SMT.y_logic(1,:);
c2 = SMT.y_logic(2,:);
lp = length(find(c1));
lq = length(find(c2));

div = std(SMT.x, 0, 2);
iConst = find(div==0);
div(iConst) = 1;
rval = ((mean(SMT.x(:,c1), 2) - mean(SMT.x(:,c2),2)) * sqrt(lp*lq)) ./ (div * (lp+lq));
rval(iConst) = NaN;
rval = reshape(rval, [time channels 1]);

[SMT.y_class{:}] = deal('sgnr^2');
SMT.class = {'1', 'sgnr^2'};
SMT.y_logic = true(1, length(SMT.y_logic));
SMT.y_dec = ones(1, length(SMT.y_dec));

out = SMT;
out.x = rval;
end