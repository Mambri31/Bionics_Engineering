%
% Questo programma calcola il Periodogramma per segnale Chirp lineare

clear all;
close all;

N=1000;
Nfft=4*1024;
f0=0.05;
f1=0.2;
f2=0.2;
A1=1;
A2=1;
Theta1=0;
Theta2=pi/8;
SNR1db=100; 

n=[0:N-1]';
k1=(f1-f0)/2/(N-1);
k2=(f2-f0)/2/(N-1);
s1=A1*cos(2*pi*(f0+k1*n).*n+Theta1);
s2=A2*cos(2*pi*(f2-k2*n).*n+Theta2);
varW=(A1^2)/2*10^(-SNR1db/10);
w=sqrt(varW)*randn(N,1);
Df=1/Nfft;
f=[-0.5:Df:0.5-Df];
x=s1+s2+w;
X=fftshift(fft(x,Nfft));
Period=((abs(X)).^2)/N;
PerioddB=10*log10(Period);


figure(1);
xlabel('Discrete Time [n]');
ylabel('X[n]');
grid on;
hold on
plot(n,x,'b','LineWidth',2);
title('Discrete Time Signal X[n] - Two Linear Chirps');
legend('X[n]');

figure(2);
xlabel('Digital Frequency (f)');
ylabel('PSD [dB]');
grid on;
hold on
plot(f,PerioddB,'b','LineWidth',2);
title('Periodogram - Two Linear Chirps');
legend('Periodogram - Two Linear Chirps');
axis([0 0.5 -50 20])


figure(3);
[y,F,n,p] = spectrogram(x,200,200-4,f,1,'yaxis'); 
surf(n,f,10*log10(abs(p)),'EdgeColor','none');   
axis xy; axis tight; colormap(jet); view(0,90);
xlabel('Discrete Time [n]'); 
ylabel('Digital Frequency [f]');
axis([0 N 0 0.5])
title('Spectrogram - Two Linear Chirps');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
