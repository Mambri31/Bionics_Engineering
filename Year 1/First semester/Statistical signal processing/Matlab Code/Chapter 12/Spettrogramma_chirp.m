%
% Questo programma calcola il Periodogramma per segnale Chirp lineare

clear all;
close all;

N=2*500;
Nfft=256;
f0=0.05;
f1=0.2;
A0=1;
Theta0=0;
SNRdb=100; 

n=[0:N-1]';
k=(f1-f0)/2/(N-1);
s=A0*cos(2*pi*(f0+k*n).*n+Theta0);
varW=(A0^2)/2*10^(-SNRdb/10);
w=sqrt(varW)*randn(N,1);
Df=1/Nfft;
f=[-0.5:Df:0.5-Df];
x=s+w;
X=fftshift(fft(x,Nfft));
Period=((abs(X)).^2)/N;
PerioddB=10*log10(Period);


figure(1);
xlabel('Discrete Time [n]');
ylabel('X[n]');
grid on;
hold on
plot(n,x,'b','LineWidth',2);
title('Discrete Time Signal X[n] - Linear Chirp');
legend('X[n]');

figure(2);
xlabel('Digital Frequency (f)');
ylabel('PSD [dB]');
grid on;
hold on
plot(f,PerioddB,'b','LineWidth',2);
title('Periodogram - Linear Chirp');
legend('Periodogram - Linear Chirp');
axis([0 0.5 -50 20])


figure(3);
[y,F,n,p] = spectrogram(x,200,200-4,f,1,'yaxis'); 
surf(n,f,10*log10(abs(p)),'EdgeColor','none');   
axis xy; axis tight; colormap(jet); view(0,90);
xlabel('Discrete Time [n]'); 
ylabel('Digital Frequency [f]');
axis([0 N 0 0.5])
title('Spectrogram - Linear Chirp');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
