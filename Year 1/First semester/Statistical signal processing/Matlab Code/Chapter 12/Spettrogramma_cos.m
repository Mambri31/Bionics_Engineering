%
% Questo programma calcola il Periodogramma per una combinazione lineare di
% due cosinusoidi
%

clear all;
close all;

N=500;
Nfft=4096;
f1=0.1;
f2=0.3;
A1=1;
A2=1;
Theta1=0;
Theta2=pi/2;
SNR1db=100; 

n=[0:N-1]';
s1=zeros(N,1);
s2=zeros(N,1);
cos1=A1*cos(2*pi*f1*n+Theta1);
cos2=A2*cos(2*pi*f2*n+Theta2);
varW=(A1^2)/2*10^(-SNR1db/10);
w=sqrt(varW)*randn(N,1);
Df=1/Nfft;
f=[-0.5:Df:0.5-Df];

% Case Study 1: cos sovrapposte temporalmente

s1(1:N/2)=cos1(1:N/2);
s2(1:N/2)=cos2(1:N/2);

x1=s1+s2+w;
X1=fftshift(fft(x1,Nfft));
Period=((abs(X1)).^2)/N;
PerioddB=10*log10(Period);

figure(1);
xlabel('Discrete Time [n]');
ylabel('X[n]');
grid on;
hold on
plot(n,x1,'b','LineWidth',1);
title('Discrete Time Signal X[n] - Case Study 1');
legend('X[n]');

figure(2);
xlabel('Digital Frequency (f)');
ylabel('PSD');
grid on;
hold on
plot(f,PerioddB,'b','LineWidth',1);
title('Periodogram - Case Study 1');
legend('Periodogram');
axis([0 0.5 -50 20])

% Case Study 2: cos NON sovrapposte temporalmente

s1=zeros(N,1);
s2=zeros(N,1);
s1(1:N/2)=cos1(1:N/2);
s2(N/2+1:N)=cos2(N/2+1:N);
x2=s1+s2+w;
X2=fftshift(fft(x2,Nfft));
Period=((abs(X2)).^2)/N;
PerioddB=10*log10(Period);

figure(3);
xlabel('Discrete Time [n]');
ylabel('X[n]');
grid on;
hold on
plot(n,x2,'b','LineWidth',1);
title('Discrete Time Signal X[n] - Case Study 2');
legend('X[n]');
% axis([0 0.5 -50 20])

figure(4);
xlabel('Digital Frequency (f)');
ylabel('PSD');
grid on;
hold on
plot(f,PerioddB,'b','LineWidth',1);
title('Periodogram - Case Study 2');
legend('Periodogram');
axis([0 0.5 -50 20])

figure(5);
[y,F,n,p] = spectrogram(x1,50,50-4,f,1,'yaxis'); 
surf(n,f,10*log10(abs(p)),'EdgeColor','none');   
axis xy; axis tight; colormap(jet); view(0,90);
xlabel('Discrete Time [n]'); 
ylabel('Digital Frequency [f]');
axis([0 N 0 0.5])
title('Spectrogram - Case Study 1');

figure(6);
[y,F,n,p] = spectrogram(x2,50,50-4,f,1,'yaxis'); 
surf(n,f,10*log10(abs(p)),'EdgeColor','none');   
axis xy; axis tight; colormap(jet); view(0,90);
xlabel('Discrete Time [n]'); 
ylabel('Digital Frequency [f]');
axis([0 N 0 0.5])
title('Spectrogram - Case Study 2');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
