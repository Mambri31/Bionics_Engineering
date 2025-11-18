%
% Questo programma calcola e disegna la funzione di autocorrelazione e la
% PSD di un processo MA(1)
%
clear all;
close all;

b1=0.98;
varX=1;
varW=varX/(1+b1^2)
N=100;
timeindex=[0:N-1];
w=sqrt(varW)*randn(1,N);
x=filter([1 b1]',1,w);

timelag=[0:10];
Rx=zeros(length(timelag),1);
Rx(1)=varX;
Rx(2)=b1*varW;
Nf=1024;
deltaf=1/Nf;
f=[0:deltaf:0.5-deltaf];
Sx=varW.*(1+b1^2+2*b1.*cos(2*pi*f));   

% Figure of the process realizations

figure(1)
stem(timeindex,x,'Linewidth', 2);grid on;
legend('Gaussian MA(1) process');
axis([0 N -3*sqrt(varX) 3*sqrt(varX)]); 
xlabel('Discrete-Time[n]')
ylabel('X[n]')
title('Gaussian MA(1) process')

% Figure of the correlation

figure(2)
stem(timelag,Rx, 'Linewidth', 2);grid on;
axis([0 10 min(Rx) max(Rx)]); 
legend('ACF di X[n]');
xlabel('Time-lag[m]')
ylabel('Funzione di Autocorrelazione (ACF)')
title('Gaussian MA(1) process')
grid on;

% Figure of PSD in linear scale

figure(3)
plot(f,Sx, 'Linewidth', 2);grid on;
legend('PSD di X[n] in scala lineare');
axis([0 0.5 0 2]); 
xlabel('Digital frequency (f)')
ylabel('Power Spectral Density (PSD)')
title('Gaussian MA(1) process')
grid on

% Figure of PSD in dB

SxdB=10*log10(Sx);

figure(4)
plot(f,SxdB, 'Linewidth', 2);grid on;
legend('PSD di X[n] in dB');
axis([0 0.5 -40 10]); 
xlabel('Digital frequency (f)')
ylabel('Power Spectral Density (PSD) [dB]')
title('Gaussian MA(1) process')
grid on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
