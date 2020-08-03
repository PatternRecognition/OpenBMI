function [output]=targetsequencer
exitflag =0;
targetSequence=1:1:12;
while exitflag ~=1
    targetSequence([1:2:12])=randperm(6);
    targetSequence([2:2:12])=randperm(6);
    for t=1:11
        if targetSequence(t)==targetSequence(t+1)
            exitflag=0;
            break;
        else
            exitflag=1;
        end
    end
end
output=targetSequence;


