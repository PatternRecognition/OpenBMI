
fs = 1000;
t = [0:1/fs:2-(1/fs)];
noise = randn(1,length(t));
x_1 = 2*sin((20*2*pi)*t)+noise;
x_2 = 10*sin((10*2*pi)*t)+noise;
x_3 = 2*sin((5*2*pi)*t)+noise;

t_2 = [0:1/fs:8-(1/fs)];
x = [noise x_1 x_2 x_3];

b=1;

ny_freq = fs/2;
vec_freq = [5 10 20];

for i=1:length(vec_freq)
  
low_freq = (vec_freq(i)-1)/ny_freq;
high_freq = (vec_freq(i)+1)/ny_freq;
f = [0 low_freq-(0.1/ny_freq) low_freq high_freq high_freq+(0.1/ny_freq) 1];
m = [0 0 1 1 0 0];
a = fir2(1600,f,m);
y(i,:) = filter(a,b,x);
Y(i,:) = abs(fft(y(i,:))); 
end

figure(1)
subplot(3,2,1)
plot(t_2,y(1,:))
subplot(3,2,3)
plot(t_2,y(2,:))
subplot(3,2,5)
plot(t_2,y(3,:))

f =((0:size(Y,2)-1)/size(Y,2))*fs;     % Omskalering fra indeksværdier til Hz.

subplot(3,2,2)
plot(f,Y(1,:))
xlim([0 50])
ylim([0 max(max(Y))])
subplot(3,2,4)
plot(f,Y(2,:))
xlim([0 50])
ylim([0 max(max(Y))])
subplot(3,2,6)
plot(f,Y(3,:))
xlim([0 50])
ylim([0 max(max(Y))])
figure(2)
subplot(1,2,1)
plot(t_2,x)
X = abs(fft(x));            
subplot(1,2,2)
plot(f,X)

xlim([0 50])