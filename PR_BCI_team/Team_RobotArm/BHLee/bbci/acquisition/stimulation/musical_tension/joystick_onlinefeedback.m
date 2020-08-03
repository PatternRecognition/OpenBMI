dat=zeros(1,1000);
%%2. variante online feedback, geht
%figure

for i= 1:3
   dat=zeros(1,1000);
   for k=1:1000
    a=jst;
    dat(k)=(a(2)+0.9379)*0.5;
    %dat=[a(2) dat(1:end-1)]
    plot([1:1000],dat)
   set(gca, 'YLim',[-0.05 1.05])
    xlabel ('[s]')
    pause(0.01)
   end
end

